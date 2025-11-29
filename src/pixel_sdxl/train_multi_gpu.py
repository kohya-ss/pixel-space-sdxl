import argparse
from datetime import datetime
from multiprocessing import Value
import os
from typing import Any, Optional
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
from pixel_sdxl.train import (
    NUM_TRAIN_TIMESTEPS,
    add_noise_x0,
    maybe_build_lpips,
    prepare_conditioning,
    prepare_optimizer,
    sample_and_save_images,
    sample_t,
    sample_t_uniform,
    save_state_to_checkpoint,
    setup_argparser,
    velocity_loss_x_pred,
)


def parse_args():
    parser = setup_argparser()
    parser.add_argument("--main_device", type=int, default=0, help="Main GPU device ID.")
    parser.add_argument("--secondary_device", type=int, default=1, help="Secondary GPU device ID.")
    return parser.parse_args()


class EncodingImageDataset(ImageDataset):
    def __init__(
        self,
        text_encoder1: nn.Module,
        text_encoder2: nn.Module,
        current_epoch: Any,
        metadata_files: str,
        batch_size: int,
        tokenizer1,
        tokenizer2,
        seed: Optional[int] = None,
        is_skipping: Optional[Any] = None,
    ):
        super().__init__(current_epoch, metadata_files, batch_size, tokenizer1, tokenizer2, seed, is_skipping)
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2

    def __getitem__(self, index):
        images, input_ids1, input_ids2, original_sizes, crop_sizes, target_sizes = super().__getitem__(index)
        if self.is_skipping is not None and self.is_skipping.value:
            return images, torch.empty((1, 77, 1)), torch.empty((1, 77, 1)), original_sizes, crop_sizes, target_sizes

        with torch.no_grad():
            context, y = prepare_conditioning(
                input_ids1,
                input_ids2,
                original_sizes,
                crop_sizes,
                target_sizes,
                self.text_encoder1,
                self.text_encoder2,
                dtype=self.text_encoder1.dtype,
                device=self.text_encoder1.device,
            )
        context = context.cpu()
        y = y.cpu()
        return images, context, y, original_sizes, crop_sizes, target_sizes


def main():
    args = parse_args()
    assert torch.cuda.device_count() >= 2, "This script requires at least 2 GPUs."
    assert args.lr_text_encoder1 == 0 and args.lr_text_encoder2 == 0, "Fine-tuning text encoders is not supported in this script."

    device_main = torch.device(f"cuda:{args.main_device}")
    device_secondary = torch.device(f"cuda:{args.secondary_device}")
    mixed_precision = not args.no_mixed_precision and device_main.type == "cuda"
    autocast_dtype = torch.float16 if mixed_precision else None

    tokenizer1, tokenizer2 = get_sdxl_tokenizers()

    print("Loading models from checkpoint...")

    # Load the state dict
    # state_dict = safetensors.torch.load(open(args.checkpoint, "rb").read())
    if args.checkpoint.endswith(".safetensors"):
        state_dict = safetensors.torch.load(open(args.checkpoint, "rb").read())
    else:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    text_encoder1, text_encoder2, unet, logit_scale = load_models_from_state_dict(
        state_dict, base_resolution=args.base_resolution, encoder_decoder_architecture=args.encoder_decoder_architecture
    )
    text_encoder1.to(device_secondary)
    text_encoder2.to(device_secondary)

    unet.to(device_main)
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

    lpips_model = maybe_build_lpips(args.lpips_lambda, torch.float32, device_secondary)

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

    current_epoch = Value("i", 0)  # shared value for epoch count across workers
    is_skipping = Value("b", False)  # shared value to indicate skipping state
    dataset = EncodingImageDataset(
        text_encoder1,
        text_encoder2,
        current_epoch,
        args.metadata_files,
        args.batch_size,
        tokenizer1,
        tokenizer2,
        seed=args.seed,
        is_skipping=is_skipping,
    )

    # dataloader with batch_size=1 since ImageDataset already returns a batch
    print("Preparing data loader... Note: num_workers is set to 1 because text encoders are used in dataset.")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)

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

    initial_steps = args.initial_steps
    if initial_steps > 0:
        print(f"Skipping to initial step {args.initial_steps}...")
        while initial_steps - step > len(data_loader):
            step += len(data_loader)
            epoch += 1
            pbar.update(len(data_loader))

        global_step += step // args.grad_accum_steps
        print(f"Continuing skipping for additional {initial_steps - step} steps in the next epoch...")
    if step < initial_steps:
        is_skipping.value = True

    try:
        while step < args.steps:
            print(f"Starting epoch {epoch+1}...")
            current_epoch.value = epoch + 1

            epoch_losses = []
            epoch += 1

            for data in data_loader:
                if step < initial_steps:
                    if (step + 1) % args.grad_accum_steps == 0:
                        global_step += 1

                    step += 1
                    pbar.update(1)
                    if step >= args.steps:
                        break

                    # stop skipping before reaching initial_steps, 20 is arbitrary to avoid prefetch issues
                    if step >= initial_steps - 20 and is_skipping.value:
                        is_skipping.value = False
                        print("Resuming training now.")
                    continue

                # Original: images, input_ids1, input_ids2, original_sizes, crop_sizes, target_sizes = data
                # With EncodingImageDataset, context and y are precomputed
                images = data[0].squeeze(0)  # B,C,H,W
                context = data[1].squeeze(0)  # B,77,dim
                y = data[2].squeeze(0)  # B,77,dim
                original_sizes = data[3].squeeze(0)  # B,2
                crop_sizes = data[4].squeeze(0)  # B,2
                target_sizes = data[5].squeeze(0)  # B,2
                assert (
                    original_sizes[0][0].item() > 0 and original_sizes[0][1].item() > 0
                ), "Illegal original size 0, `20` may be too small for pre-fetching issues."

                images = images.to(device=device_main, dtype=unet_dtype)
                context = context.to(device=device_main, dtype=unet_dtype)
                y = y.to(device=device_main, dtype=unet_dtype)
                original_sizes = original_sizes.to(device=device_main)
                crop_sizes = crop_sizes.to(device=device_main)
                target_sizes = target_sizes.to(device=device_main)

                # 事前にエンコードされた条件付けを使用

                B = images.shape[0]
                x0 = images

                # 1. t を logit-normal からサンプリング (0: clean, 1: noise)
                if args.uniform_sampling:
                    t = sample_t_uniform(B, device=device_main)  # [B]
                else:
                    t = sample_t(B, device=device_main, mu=mu, sigma=sigma)  # [B]

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

                    # LPIPS 勾配を後で追加するために勾配を保持
                    if lpips_model is not None:
                        x0_pred_resized.retain_grad()

                    # [-1,1] で LPIPS を計算（出力をクランプ）
                    lpips_pred = torch.clamp(x0_pred_resized, -1.0, 1.0).float()
                    lpips_target = torch.clamp(x0_resized, -1.0, 1.0).float()

                    # device_secondary で勾配計算用のテンソルを作成
                    lpips_pred_secondary = lpips_pred.detach().to(device_secondary).requires_grad_(True)
                    lpips_target_secondary = lpips_target.detach().to(device_secondary)

                    # LPIPS loss を計算
                    lpips_out = lpips_model(lpips_pred_secondary, lpips_target_secondary)
                    lpips_out = lpips_out * (1 - t.to(device_secondary)).view(B, 1, 1, 1)
                    lpips_loss_val = lpips_out.mean()

                    # device_secondary で勾配を計算
                    lpips_loss_val.backward()
                    lpips_grad = lpips_pred_secondary.grad.to(device_main)

                    # ログ用に値を保持（勾配グラフから切り離す）
                    lpips_loss_val = lpips_loss_val.detach().to(device_main)

                total_loss = base_loss
                loss = total_loss / args.grad_accum_steps

                if mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # LPIPS の勾配を x0_pred.grad に追加
                if lpips_model is not None and x0_pred_resized.grad is not None:
                    # クランプの勾配を考慮（-1 < x0_pred < 1 の範囲のみ勾配を流す）
                    clamp_mask = ((x0_pred_resized.detach() > -1.0) & (x0_pred_resized.detach() < 1.0)).float()
                    scaled_lpips_grad = args.lpips_lambda * lpips_grad * clamp_mask / args.grad_accum_steps
                    x0_pred_resized.grad.add_(scaled_lpips_grad)

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
