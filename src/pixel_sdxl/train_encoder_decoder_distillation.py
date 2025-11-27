import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Value
import os
from pixel_sdxl import sdxl_original_unet, vae_utils
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import bitsandbytes as bnb
from tqdm import tqdm
import safetensors.torch

from pixel_sdxl.image_dataset import ImageDataset
from pixel_sdxl.model_utils import load_models_from_state_dict, serialize_models_to_state_dict
from pixel_sdxl import train_utils
from pixel_sdxl.text_encoder_utils import encode_tokens, get_sdxl_tokenizers
from pixel_sdxl.sdxl_pixel_unet import SDXLPixelUNet


NUM_TRAIN_TIMESTEPS = 1000
COMPILE_MODEL = True


def prepare_optimizer(
    unet: SDXLPixelUNet,
    lr_main: float,
    train_encoder: bool,
    train_decoder: bool,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    param_groups: list[dict] = []

    def maybe_add_group(params, lr: float, name: str):
        params = [p for p in params if p.requires_grad]
        if lr > 0 and params:
            param_groups.append({"params": params, "lr": lr})
        elif lr == 0:
            for p in params:
                p.requires_grad_(False)
            text = f"{name} is frozen (lr=0)"
            print(text)

    encoder_params = [p for n, p in unet.named_parameters() if "encoder" in n]
    decoder_params = [p for n, p in unet.named_parameters() if "decoder" in n]

    maybe_add_group(encoder_params, lr_main if train_encoder else 0, "Encoder")
    maybe_add_group(decoder_params, lr_main if train_decoder else 0, "Decoder")
    num_params = sum(p.numel() for group in param_groups for p in group["params"])
    print(f"Total trainable parameters: {num_params / 1e6:.2f} million")

    if not param_groups:
        raise ValueError("No trainable parameters. Please set at least one learning rate > 0.")

    # optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=weight_decay)
    optimizer = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.999), weight_decay=weight_decay)
    return optimizer


def prepare_conditioning(
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    original_sizes: torch.Tensor,
    crop_top_lefts: torch.Tensor,
    target_sizes: torch.Tensor,
    text_encoder1: nn.Module,
    text_encoder2: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
):
    hidden_states1, hidden_states2, pooled2 = encode_tokens(input_ids1, input_ids2, text_encoder1, text_encoder2)
    context = torch.cat([hidden_states1, hidden_states2], dim=-1).to(device=device, dtype=dtype)
    size_embeddings = train_utils.get_size_embeddings(original_sizes, crop_top_lefts, target_sizes, device=device).to(dtype)
    y = torch.cat([pooled2.to(device=device, dtype=dtype), size_embeddings], dim=-1)
    return context, y


def parse_args():
    parser = argparse.ArgumentParser(description="Train distillation pixel-space SDXL Encoder/Decoder.")
    parser.add_argument("--device1", type=str, default="cuda:0", help="Device 1 for training (default: cuda:0).")
    parser.add_argument("--device2", type=str, default="cuda:0", help="Device 2 for training (default: cuda:0).")
    parser.add_argument(
        "--base_resolution", type=int, choices=[32, 64], default=64, help="Base resolution for Pixel U-Net (32 or 64)."
    )
    parser.add_argument("--metadata_files", nargs="+", required=True, help="List of metadata JSON files.")
    parser.add_argument("--checkpoint", required=True, help="Path to a safetensors checkpoint to load weights from.")
    parser.add_argument(
        "--teacher_checkpoint", type=str, required=True, help="Path to a safetensors checkpoint for the teacher model."
    )
    parser.add_argument("--save_path", required=True, help="Path to save the updated checkpoint.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for ImageDataset.")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimization steps.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Encoder/Decoder.")
    parser.add_argument(
        "--train_encoder", action="store_true", help="Train the Encoder model, if not specified, Decoder is trained."
    )
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm. Set 0 to disable.")
    parser.add_argument("--uniform_sampling", action="store_true", help="Use uniform time sampling instead of logit-normal.")
    parser.add_argument("--save_interval", type=int, default=0, help="Save every N steps (0 means save only at the end).")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable autocast/GradScaler.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing on U-Net.")
    parser.add_argument("--compile_model", action="store_true", help="Compile the U-Net model with torch.compile().")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def sample_t(batch_size, device, mu=0.8, sigma=0.8):
    # s ~ N(mu, sigma^2)
    s = torch.randn(batch_size, device=device) * sigma + mu  # [B]
    t = torch.sigmoid(s)  # (0, 1)  にマップ
    return t  # shape [B]


def sample_t_uniform(batch_size, device):
    t = torch.rand(batch_size, device=device)  # [B], uniform [0,1)
    return t


def add_noise_x0(x0, t, noise_scale=1.0):
    """
    x0: [B, 3, H, W]
    t:  [B] in [0, 1]  (0: clean, 1: noise)
    """
    B = x0.size(0)

    eps = torch.randn_like(x0) * noise_scale  # ϵ ~ N(0, noise_scale^2 I)

    # shape を [B,1,1,1] に伸ばしてブロードキャスト
    t_ = t.view(B, 1, 1, 1)

    x_t = (1.0 - t_) * x0 + t_ * eps
    return x_t, eps


def velocity_loss_x_pred(x0, x_t, x0_pred, t, t_clip=0.05):
    """
    x0:      clean image      [B,3,H,W]
    x_t:     noised image     [B,3,H,W]
    x0_pred: model output     [B,3,H,W]
    t:       time in [0,1]    [B]
    """

    B = x0.size(0)
    t_ = t.view(B, 1, 1, 1)
    denom = torch.clamp(t_, min=t_clip)  # 0付近で暴れないように

    v = (x0 - x_t) / denom
    v_pred = (x0_pred - x_t) / denom

    loss = F.mse_loss(v_pred, v)
    return loss


def save_state_to_checkpoint(
    ckpt_path: str,
    unet: SDXLPixelUNet,
    text_encoder1: nn.Module,
    text_encoder2: nn.Module,
    optimizer: torch.optim.Optimizer,
    logit_scale: torch.Tensor,
):
    state_dict = {}
    state_dict["model"] = serialize_models_to_state_dict(text_encoder1, text_encoder2, unet, logit_scale)
    state_dict["optimizer_state"] = optimizer.state_dict()
    torch.save(state_dict, ckpt_path)


def get_teacher_features(train_encoder, vae, unet_teacher, images, t, context, y):
    with torch.no_grad(), autocast(device_type=vae.device.type, enabled=False):
        # VAEで潜在変数にエンコード
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor  # [B,4,H/8,W/8]

    with torch.no_grad(), autocast(device_type=unet_teacher.device.type, enabled=True):
        # 教師U-Netで特徴量を取得
        teacher_output = unet_teacher(
            x=latents,
            timesteps=t * (NUM_TRAIN_TIMESTEPS - 1),
            context=context,
            y=y,
            return_encoder_features=train_encoder,
            return_decoder_features=not train_encoder,
        )
    return teacher_output  #  feature map


def main():
    args = parse_args()

    device1 = torch.device(args.device1)  # training U-Net encoder/decoder
    device2 = torch.device(args.device2)  # get teacher features

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = not args.no_mixed_precision and device1.type == "cuda"
    autocast_dtype = torch.float16 if mixed_precision else None

    tokenizer1, tokenizer2 = get_sdxl_tokenizers()
    current_epoch = Value("i", 0)  # shared value for epoch count across workers
    dataset = ImageDataset(current_epoch, args.metadata_files, args.batch_size, tokenizer1, tokenizer2, seed=args.seed)

    # dataloader with batch_size=1 since ImageDataset already returns a batch
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    print("Loading models from checkpoint...")

    # Load the state dict
    # state_dict = safetensors.torch.load(open(args.checkpoint, "rb").read())
    if args.checkpoint.endswith(".safetensors"):
        state_dict = safetensors.torch.load(open(args.checkpoint, "rb").read())
    else:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    text_encoder1, text_encoder2, unet, logit_scale = load_models_from_state_dict(state_dict, base_resolution=args.base_resolution)
    text_encoder1.to(device1)
    text_encoder2.to(device1)

    unet.to(device1)
    print("Models loaded. Casting U-Net to float32 for stable training...")
    unet.to(torch.float32)  # ensure float32 for stable training even with mixed precision

    text_encoder1.eval()
    text_encoder2.eval()
    unet.train()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for U-Net.")

    if args.compile_model:
        print("Compiling U-Net model with torch.compile()...")
        unet = torch.compile(unet, fullgraph=True, dynamic=True)
        print("U-Net model compiled.")

    # teacher model
    if args.teacher_checkpoint.endswith(".safetensors"):
        teacher_state_dict = safetensors.torch.load(open(args.teacher_checkpoint, "rb").read())
    else:
        teacher_state_dict = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)

    vae = vae_utils.load_vae_from_state_dict(teacher_state_dict)
    vae.to(device2, dtype=torch.float32)  # SDXL VAEはfloat32で安定動作
    vae.eval()

    unet_teacher = sdxl_original_unet.SdxlUNet2DConditionModel()
    unet_sd = {
        k[len("model.diffusion_model.") :]: v for k, v in teacher_state_dict.items() if k.startswith("model.diffusion_model.")
    }
    unet_teacher.load_state_dict(unet_sd, strict=True, assign=True)
    unet_teacher.to(device2)
    unet_teacher.to(autocast_dtype if mixed_precision else torch.float32)
    unet_teacher.eval()
    print("Teacher models loaded.")
    del teacher_state_dict

    optimizer = prepare_optimizer(unet, args.lr, args.train_encoder, not args.train_encoder)

    if "optimizer_state" in state_dict:
        print("Loading optimizer state from checkpoint...")
        optimizer.load_state_dict(state_dict["optimizer_state"])

    mu = 0.8
    sigma = 0.8
    # t_eps = 5e-2

    scaler = GradScaler(enabled=mixed_precision)
    unet_dtype = next(unet.parameters()).dtype
    clip_params = [p for p in unet.parameters() if p.requires_grad]

    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard writer
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_name = os.path.splitext(os.path.basename(args.save_path))[0]
    writer = SummaryWriter(log_dir=f"logs/{timestamp_str}_{session_name}")

    global_step = 0
    step = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(args.steps))
    accum_loss = 0.0  # gradient accumulation中のloss合計

    executor = None if device1 == device2 else ThreadPoolExecutor(max_workers=1)

    while step < args.steps:
        print(f"Starting epoch {epoch+1}...")
        current_epoch.value = epoch + 1

        epoch_losses = []
        epoch += 1

        for data in data_loader:
            # images, input_ids1, input_ids2, original_sizes, crop_sizes, target_sizes = data
            images = data[0].squeeze(0)  # B,C,H,W
            input_ids1 = data[1].squeeze(0)  # B,77
            input_ids2 = data[2].squeeze(0)  # B,77
            original_sizes = data[3].squeeze(0)  # B,2
            crop_sizes = data[4].squeeze(0)  # B,2
            target_sizes = data[5].squeeze(0)  # B,2

            images = images.to(device=device1, dtype=unet_dtype)
            input_ids1 = input_ids1.to(device1)
            input_ids2 = input_ids2.to(device1)
            original_sizes = original_sizes.to(device1)
            crop_sizes = crop_sizes.to(device1)
            target_sizes = target_sizes.to(device1)

            with autocast(device_type="cuda", dtype=autocast_dtype, enabled=mixed_precision):
                context, y = prepare_conditioning(
                    input_ids1,
                    input_ids2,
                    original_sizes,
                    crop_sizes,
                    target_sizes,
                    text_encoder1,
                    text_encoder2,
                    dtype=unet_dtype,
                    device=device1,
                )

            B = images.shape[0]
            x0 = images

            # 1. t を logit-normal からサンプリング (0: clean, 1: noise)
            if args.uniform_sampling:
                t = sample_t_uniform(B, device=device1)  # [B]
            else:
                t = sample_t(B, device=device1, mu=mu, sigma=sigma)  # [B]

            # 教師モデルから特徴量を取得
            if executor is not None:
                images_device2 = images.to(device=device2, dtype=torch.float32)
                t_device2 = t.to(device=device2)
                context_device2 = context.to(device=device2)
                y_device2 = y.to(device=device2)
                future = executor.submit(
                    get_teacher_features,
                    args.train_encoder,
                    vae,
                    unet_teacher,
                    images_device2,
                    t_device2,
                    context_device2,
                    y_device2,
                )
                teacher_features = None
            else:
                future = None
                teacher_features = get_teacher_features(args.train_encoder, vae, unet_teacher, images, t, context, y)

            # 2. ノイズ混入
            x_t, eps = add_noise_x0(x0, t, noise_scale=1.0)  # noise_scale)  # [B,3,H,W]

            # 3. モデル呼び出し（x-prediction）
            # x0_pred = forward_model(model, x_t, t, text_emb)
            with autocast(device_type="cuda", dtype=autocast_dtype, enabled=mixed_precision):
                if args.train_encoder:
                    features = unet.call_encoder(x_t)
                    if future is not None:
                        teacher_features = future.result()
                        teacher_features = teacher_features.to(device=device1, dtype=unet_dtype)

                    # 4. loss計算とbackward
                    loss = F.mse_loss(features, teacher_features) / args.grad_accum_steps
                else:
                    if future is not None:
                        teacher_features = future.result()
                        teacher_features = teacher_features.to(device=device1, dtype=unet_dtype)
                    features = unet.call_decoder(teacher_features)

                    # 4. loss計算とbackward
                    loss = F.mse_loss(features, x0) / args.grad_accum_steps

            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # gradient accumulation中のlossを蓄積
            accum_loss += loss.item() * args.grad_accum_steps

            # epoch平均loss計算用に保存
            epoch_losses.append(loss.item() * args.grad_accum_steps)

            if (step + 1) % args.grad_accum_steps == 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, args.clip_grad_norm)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # gradient accumulation全体の平均lossを計算
                avg_accum_loss = accum_loss / args.grad_accum_steps
                pbar.set_description(f"loss={avg_accum_loss:.4f}")

                # TensorBoardにgradient accumulation全体の平均lossを記録
                writer.add_scalar("train/loss_step", avg_accum_loss, global_step)

                accum_loss = 0.0  # リセット

            if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                save_path_body, split_ext = os.path.splitext(args.save_path)
                save_path = f"{save_path_body}_step{step + 1}{split_ext}"
                save_state_to_checkpoint(save_path, unet, text_encoder1, text_encoder2, optimizer, logit_scale)
                print(f"Checkpoint saved at step {step + 1}: {save_path}")

            step += 1
            pbar.update(1)

            if step >= args.steps:
                break

        # epochごとの平均lossを計算してTensorBoardに記録
        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch)
            print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

    print("Saving final checkpoint...")
    save_state_to_checkpoint(args.save_path, unet, text_encoder1, text_encoder2, optimizer, logit_scale)
    print(f"Training finished. Saved to {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()
