import argparse
from datetime import datetime
from multiprocessing import Value
import os
from typing import Optional
from pixel_sdxl import sdxl_pixel_unet
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
from pixel_sdxl.inference import generate_image, InferenceConfig, tensor_to_pil_image


NUM_TRAIN_TIMESTEPS = 1000
COMPILE_MODEL = True


def prepare_optimizer(
    unet: SDXLPixelUNet,
    text_encoder1: nn.Module,
    text_encoder2: nn.Module,
    lr_main: float,
    lr_unet_body: float,
    lr_text_encoder1: float,
    lr_text_encoder2: float,
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

    unet_body_params = [p for n, p in unet.named_parameters() if unet.is_unet_body_parameter(n)]
    other_unet_params = [p for n, p in unet.named_parameters() if not unet.is_unet_body_parameter(n)]

    maybe_add_group(other_unet_params, lr_main, "Pixel U-Net (patch/decoder)")
    maybe_add_group(unet_body_params, lr_unet_body, "SDXLUNetBody")
    maybe_add_group(text_encoder1.parameters(), lr_text_encoder1, "Text Encoder 1")
    maybe_add_group(text_encoder2.parameters(), lr_text_encoder2, "Text Encoder 2")

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


def maybe_build_lpips(weight: float, dtype: torch.dtype, device: torch.device):
    if weight <= 0:
        return None
    try:
        import lpips
    except ImportError as e:
        raise ImportError("`lpips` package is required when --lpips_lambda > 0.") from e
    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.to(device=device, dtype=dtype)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)
    print(f"LPIPS loss enabled (lambda={weight}, net=vgg).")
    return lpips_model


def setup_argparser():
    parser = argparse.ArgumentParser(description="Train pixel-space SDXL with x-pred and velocity loss.")
    parser.add_argument(
        "--base_resolution", type=int, choices=[32, 64], default=64, help="Base resolution for Pixel U-Net (32 or 64)."
    )
    parser.add_argument("--metadata_files", nargs="+", required=True, help="List of metadata JSON files.")
    parser.add_argument("--checkpoint", required=True, help="Path to a safetensors checkpoint to load weights from.")
    parser.add_argument(
        "--no_restore_optimizer", action="store_true", help="Do not restore the optimizer state from the checkpoint."
    )
    parser.add_argument("--save_path", required=True, help="Path to save the updated checkpoint.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for ImageDataset.")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimization steps.")
    parser.add_argument("--initial_steps", type=int, default=0, help="Initial steps already done (skip to this step).")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Pixel U-Net parts except SDXLUNetBody.")
    parser.add_argument("--lr_sdxl_unet_body", type=float, default=1e-5, help="Learning rate for SDXLUNetBody (0 to freeze).")
    parser.add_argument("--lr_text_encoder1", type=float, default=0.0, help="Learning rate for text encoder 1 (0 to freeze).")
    parser.add_argument("--lr_text_encoder2", type=float, default=0.0, help="Learning rate for text encoder 2 (0 to freeze).")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm. Set 0 to disable.")
    parser.add_argument("--velocity_loss", action="store_true", help="Use velocity loss instead of x-pred loss.")
    parser.add_argument(
        "--lpips_lambda",
        type=float,
        default=0.0,
        help="Weight for LPIPS perceptual loss; set 0 to disable.",
    )
    parser.add_argument("--uniform_sampling", action="store_true", help="Use uniform time sampling instead of logit-normal.")
    parser.add_argument("--save_interval", type=int, default=0, help="Save every N steps (0 means save only at the end).")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable autocast/GradScaler.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing on U-Net.")
    parser.add_argument("--compile_model", action="store_true", help="Compile the U-Net model with torch.compile().")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sample_every", type=int, default=0, help="Sample and save images every N steps (0 to disable).")
    parser.add_argument("--sample_prompts", nargs="+", default=[], help="List of prompts to use for sampling during training.")
    parser.add_argument("--sample_seed", type=int, default=1234, help="Random seed for sampling during training.")
    return parser


def parse_args():
    parser = setup_argparser()
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

    v = (x_t - x0) / denom
    v_pred = (x_t - x0_pred) / denom

    loss = F.mse_loss(v_pred, v)
    return loss


def sample_and_save_images(
    step: int,
    unet: SDXLPixelUNet,
    text_encoder1: nn.Module,
    text_encoder2: nn.Module,
    prompts: list[str],
    seed: int,
    save_dir: str,
    tokenizer1,
    tokenizer2,
    autocast_dtype: Optional[torch.dtype] = None,
):
    is_unet_train = unet.training
    is_te1_train = text_encoder1.training
    is_te2_train = text_encoder2.training
    unet.eval()
    text_encoder1.eval()
    text_encoder2.eval()

    os.makedirs(save_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        config = InferenceConfig(
            prompt=prompt,
            precision="float16" if autocast_dtype == torch.float16 else "float32",
            seed=seed + i,
        )
        image_tensor = generate_image(
            unet, text_encoder1, text_encoder2, tokenizer1, tokenizer2, config, autocast_dtype=autocast_dtype
        )
        image = tensor_to_pil_image(image_tensor)
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        image.save(os.path.join(save_dir, f"{timestamp_str}_step_{step}_sample_{i+1}.png"))

    unet.train(is_unet_train)
    text_encoder1.train(is_te1_train)
    text_encoder2.train(is_te2_train)


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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = not args.no_mixed_precision and device.type == "cuda"
    autocast_dtype = torch.float16 if mixed_precision else None

    tokenizer1, tokenizer2 = get_sdxl_tokenizers()
    current_epoch = Value("i", 0)  # shared value for epoch count across workers
    is_skipping = Value("b", False)  # shared value to indicate skipping state
    dataset = ImageDataset(
        current_epoch, args.metadata_files, args.batch_size, tokenizer1, tokenizer2, seed=args.seed, is_skipping=is_skipping
    )

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
    text_encoder1.to(device)
    text_encoder2.to(device)

    unet.to(device)
    print("Models loaded. Casting U-Net to float32 for stable training...")
    unet.to(torch.float32)  # ensure float32 for stable training even with mixed precision

    text_encoder1.train(mode=args.lr_text_encoder1 > 0)
    if args.lr_text_encoder1 > 0:
        text_encoder1.gradient_checkpointing_enable()
    text_encoder2.train(mode=args.lr_text_encoder2 > 0)
    if args.lr_text_encoder2 > 0:
        text_encoder2.gradient_checkpointing_enable()
    unet.train()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for U-Net.")

    if args.compile_model:
        print("Compiling U-Net model with torch.compile()...")
        unet = torch.compile(unet, fullgraph=True, dynamic=True)
        print("U-Net model compiled.")

    lpips_model = maybe_build_lpips(args.lpips_lambda, torch.float32, device)

    optimizer = prepare_optimizer(
        unet,
        text_encoder1,
        text_encoder2,
        lr_main=args.lr,
        lr_unet_body=args.lr_sdxl_unet_body,
        lr_text_encoder1=args.lr_text_encoder1,
        lr_text_encoder2=args.lr_text_encoder2,
    )

    if "optimizer_state" in state_dict:
        if args.no_restore_optimizer:
            print("Skipping optimizer state restoration as per --no_restore_optimizer flag.")
        else:
            print("Loading optimizer state from checkpoint...")
            lr_for_groups = [group["lr"] for group in optimizer.param_groups]
            print(f"Learning rates by args: {lr_for_groups}")
            optimizer.load_state_dict(state_dict["optimizer_state"])
            lr_for_groups_in_ckpt = [group["lr"] for group in optimizer.param_groups]
            print(f"Learning rates from checkpoint: {lr_for_groups_in_ckpt}")
            for lr, group in zip(lr_for_groups, optimizer.param_groups):
                group["lr"] = lr
            print("Optimizer state loaded and learning rates adjusted.")

    mu = 0.8
    sigma = 0.8
    # t_eps = 5e-2

    scaler = GradScaler(enabled=mixed_precision)
    unet_dtype = next(unet.parameters()).dtype
    clip_params = [p for p in unet.parameters() if p.requires_grad]
    if args.lr_text_encoder1 > 0:
        clip_params += [p for p in text_encoder1.parameters() if p.requires_grad]
    if args.lr_text_encoder2 > 0:
        clip_params += [p for p in text_encoder2.parameters() if p.requires_grad]

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
    accum_lpips_loss = 0.0  # gradient accumulation中のLPIPS loss合計

    # generate initial samples before training
    if args.sample_every > 0 and args.sample_prompts:
        print("Generating initial sample images before training...")
        sample_and_save_images(
            0,
            unet,
            text_encoder1,
            text_encoder2,
            args.sample_prompts,
            args.sample_seed,
            save_dir,
            tokenizer1,
            tokenizer2,
            autocast_dtype,
        )

    if args.initial_steps > 0:
        print(f"Skipping to initial step {args.initial_steps}...")
        is_skipping.value = True

    try:
        while step < args.steps:
            print(f"Starting epoch {epoch+1}...")
            current_epoch.value = epoch + 1

            epoch_losses = []
            epoch += 1

            for data in data_loader:
                if step < args.initial_steps:
                    if (step + 1) % args.grad_accum_steps == 0:
                        global_step += 1

                    step += 1
                    pbar.update(1)
                    if step >= args.steps:
                        break

                    # stop skipping before reaching initial_steps, 20 is arbitrary to avoid prefetch issues
                    if step >= args.initial_steps - 20 and is_skipping.value:
                        is_skipping.value = False
                        print("Resuming training now.")
                    continue

                # images, input_ids1, input_ids2, original_sizes, crop_sizes, target_sizes = data
                images = data[0].squeeze(0)  # B,C,H,W
                input_ids1 = data[1].squeeze(0)  # B,77
                input_ids2 = data[2].squeeze(0)  # B,77
                original_sizes = data[3].squeeze(0)  # B,2
                crop_sizes = data[4].squeeze(0)  # B,2
                target_sizes = data[5].squeeze(0)  # B,2
                assert (
                    original_sizes[0][0].item() > 0 and original_sizes[0][1].item() > 0
                ), "Illegal original size 0, maybe skipped all data in this epoch."

                images = images.to(device=device, dtype=unet_dtype)
                input_ids1 = input_ids1.to(device)
                input_ids2 = input_ids2.to(device)
                original_sizes = original_sizes.to(device)
                crop_sizes = crop_sizes.to(device)
                target_sizes = target_sizes.to(device)

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
                        device=device,
                    )

                B = images.shape[0]
                x0 = images

                # 1. t を logit-normal からサンプリング (0: clean, 1: noise)
                if args.uniform_sampling:
                    t = sample_t_uniform(B, device=device)  # [B]
                else:
                    t = sample_t(B, device=device, mu=mu, sigma=sigma)  # [B]

                # 2. ノイズ混入
                x_t, eps = add_noise_x0(x0, t, noise_scale=1.0)  # noise_scale)  # [B,3,H,W]

                # 3. モデル呼び出し（x-prediction）
                with autocast(device_type="cuda", dtype=autocast_dtype, enabled=mixed_precision):
                    x0_pred = unet(
                        x_t=x_t,
                        timesteps=t * (NUM_TRAIN_TIMESTEPS - 1),
                        context=context,
                        y=y,
                    )

                    if args.velocity_loss:
                        # 4-a. v-loss 計算
                        base_loss = velocity_loss_x_pred(x0, x_t, x0_pred, t)
                    else:
                        # 4-b. x-pred loss 計算
                        base_loss = F.mse_loss(x0_pred, x0)

                lpips_loss_val = None
                if lpips_model is not None:
                    # モデル出力およびターゲット画像が1024x1024で大きいので、半分にリサイズしてLPIPS計算
                    x0_pred_resized = F.interpolate(x0_pred, scale_factor=0.5, mode="bilinear", align_corners=False)
                    x0_resized = F.interpolate(x0, scale_factor=0.5, mode="bilinear", align_corners=False)
                    
                    # [-1,1] で LPIPS を計算（出力をクランプ）
                    lpips_pred = torch.clamp(x0_pred_resized, -1.0, 1.0).float()
                    lpips_target = torch.clamp(x0_resized, -1.0, 1.0).float()
                    lpips_loss_val = lpips_model(lpips_pred, lpips_target)
                    lpips_loss_val = lpips_loss_val * (1 - t).view(B, 1, 1, 1)  # tが大きいほど（ノイズに近いほど）重みを下げる
                    lpips_loss_val = lpips_loss_val.mean()

                total_loss = base_loss
                if lpips_loss_val is not None:
                    total_loss = total_loss + args.lpips_lambda * lpips_loss_val

                loss = total_loss / args.grad_accum_steps

                if mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # gradient accumulation中のlossを蓄積
                accum_loss += loss.item() * args.grad_accum_steps  # 元のlossに戻すために*grad_accum_steps
                if lpips_loss_val is not None:
                    accum_lpips_loss += lpips_loss_val.item()  # ここはそのまま足す

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
                    if lpips_loss_val is not None:
                        avg_accum_loss_lpips = accum_lpips_loss / args.grad_accum_steps
                        avg_base_loss = avg_accum_loss - avg_accum_loss_lpips * args.lpips_lambda
                        writer.add_scalar("train/base_loss_step", avg_base_loss, global_step)
                        writer.add_scalar("train/lpips_step", avg_accum_loss_lpips, global_step)

                    accum_loss = 0.0  # リセット
                    accum_lpips_loss = 0.0  # リセット

                if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    save_path_body, split_ext = os.path.splitext(args.save_path)
                    save_path = f"{save_path_body}_step{step + 1}{split_ext}"
                    save_state_to_checkpoint(save_path, unet, text_encoder1, text_encoder2, optimizer, logit_scale)
                    print(f"Checkpoint saved at step {step + 1}: {save_path}")

                if args.sample_every > 0 and (step + 1) % args.sample_every == 0 and args.sample_prompts:
                    print(f"Generating sample images at step {step + 1}...")
                    sample_and_save_images(
                        step + 1,
                        unet,
                        text_encoder1,
                        text_encoder2,
                        args.sample_prompts,
                        args.sample_seed,
                        save_dir,
                        tokenizer1,
                        tokenizer2,
                        autocast_dtype,
                    )

                step += 1
                pbar.update(1)

                if step >= args.steps:
                    break

            # epochごとの平均lossを計算してTensorBoardに記録
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch)
                print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving current checkpoint...")
        save_path_body, split_ext = os.path.splitext(args.save_path)
        interrupted_save_path = f"{save_path_body}_interrupted_step{step}{split_ext}"
        save_state_to_checkpoint(interrupted_save_path, unet, text_encoder1, text_encoder2, optimizer, logit_scale)
        print(f"Checkpoint saved: {interrupted_save_path}")

        # close the TensorBoard writer
        writer.close()
        return

    print("Saving final checkpoint...")
    save_state_to_checkpoint(args.save_path, unet, text_encoder1, text_encoder2, optimizer, logit_scale)
    print(f"Training finished. Saved to {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()
