import argparse
from datetime import datetime
from multiprocessing import Value
import os
import signal
import threading
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import safetensors.torch
from tqdm import tqdm

from pixel_sdxl.image_dataset import ImageDataset
from pixel_sdxl.model_utils import load_models_from_state_dict
from pixel_sdxl.text_encoder_utils import get_sdxl_tokenizers
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
    velocity_loss_x_pred,
)

interrupt_event = threading.Event()


def handle_sigint(signum, frame):
    if not interrupt_event.is_set():
        interrupt_event.set()
        if dist.is_initialized() and dist.get_rank() == 0:
            print("SIGINT received. Finishing current iteration and checkpoint save...")
    else:
        print("Second SIGINT received. Exiting immediately.")
        raise KeyboardInterrupt


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training for pixel-space SDXL with x-pred and velocity loss.")
    parser.add_argument(
        "--base_resolution", type=int, choices=[32, 64], default=64, help="Base resolution for Pixel U-Net (32 or 64)."
    )
    parser.add_argument("--metadata_files", nargs="+", required=True, help="List of metadata JSON files.")
    parser.add_argument("--checkpoint", required=True, help="Path to a safetensors checkpoint to load weights from.")
    parser.add_argument(
        "--no_restore_optimizer", action="store_true", help="Do not restore the optimizer state from the checkpoint."
    )
    parser.add_argument("--save_path", required=True, help="Path to save the updated checkpoint.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU batch size for ImageDataset (dataset returns a pre-built batch).",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Total optimization steps.")
    parser.add_argument("--initial_steps", type=int, default=0, help="Initial steps already done (skip to this step).")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Pixel U-Net parts except SDXLUNetBody.",
    )
    parser.add_argument(
        "--lr_sdxl_unet_body",
        type=float,
        default=1e-5,
        help="Learning rate for SDXLUNetBody (0 to freeze).",
    )
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
    parser.add_argument(
        "--backend",
        choices=["nccl", "gloo"],
        default="nccl",
        help="Backend for torch.distributed (use gloo on Windows if nccl is unavailable).",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per rank.")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory in DataLoader.")
    return parser.parse_args()


def setup_distributed(backend: str):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank

    # Windows環境でgloo使用時に必要な環境変数を設定
    if "RANK" not in os.environ:
        # 環境変数が未設定の場合はシングルGPUモードとして動作
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        print("Warning: RANK not set. Running in single-GPU mode with RANK=0, WORLD_SIZE=1")

    dist.init_process_group(backend=backend)  # , init_method="env://?use_libuv=False")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    rank, world_size, local_rank = setup_distributed(args.backend)
    is_main_process = rank == 0

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    mixed_precision = not args.no_mixed_precision and device.type == "cuda"
    autocast_dtype = torch.float16 if mixed_precision else None

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer1, tokenizer2 = get_sdxl_tokenizers()
    current_epoch = Value("i", 0)
    is_skipping = Value("b", False)
    dataset = ImageDataset(
        current_epoch, args.metadata_files, args.batch_size, tokenizer1, tokenizer2, seed=args.seed, is_skipping=is_skipping
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=args.pin_memory,
    )

    if is_main_process:
        print("Loading models from checkpoint...")

    if args.checkpoint.endswith(".safetensors"):
        state_dict = safetensors.torch.load(open(args.checkpoint, "rb").read())
    else:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    text_encoder1, text_encoder2, unet, logit_scale = load_models_from_state_dict(state_dict, base_resolution=args.base_resolution)
    text_encoder1.to(device)
    text_encoder2.to(device)
    unet.to(device)
    if is_main_process:
        print("Models loaded. Casting U-Net to float32 for stable training...")
    unet.to(torch.float32)

    text_encoder1.train(mode=args.lr_text_encoder1 > 0)
    if args.lr_text_encoder1 > 0:
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
    else:
        text_encoder1.to(torch.float16)  # Freeze and cast to float16 to save memory

    text_encoder2.train(mode=args.lr_text_encoder2 > 0)
    if args.lr_text_encoder2 > 0:
        if args.gradient_checkpointing:
            text_encoder2.gradient_checkpointing_enable()
    else:
        text_encoder2.to(torch.float16)  # Freeze and cast to float16 to save memory

    unet.train()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if is_main_process:
            print("Gradient checkpointing enabled for U-Net.")

    if args.compile_model:
        if is_main_process:
            print("Compiling U-Net model with torch.compile()...")
        unet = torch.compile(unet, fullgraph=True, dynamic=True)
        if is_main_process:
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

    lr_for_groups = [group["lr"] for group in optimizer.param_groups]

    if "optimizer_state" in state_dict:
        if args.no_restore_optimizer:
            if is_main_process:
                print("Skipping optimizer state restoration as per --no_restore_optimizer flag.")
        else:
            if is_main_process:
                print("Loading optimizer state from checkpoint...")
                print(f"Learning rates by args: {lr_for_groups}")
            optimizer.load_state_dict(state_dict["optimizer_state"])
            if is_main_process:
                lr_for_groups_in_ckpt = [group["lr"] for group in optimizer.param_groups]
                print(f"Learning rates from checkpoint: {lr_for_groups_in_ckpt}")
            for lr, group in zip(lr_for_groups, optimizer.param_groups):
                group["lr"] = lr
            if is_main_process:
                print("Optimizer state loaded and learning rates adjusted.")

    unet_ddp = DDP(unet, device_ids=[local_rank], output_device=local_rank)
    # , broadcast_buffers=False , gradient_as_bucket_view=True)
    te1_ddp: nn.Module = (
        DDP(text_encoder1, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if args.lr_text_encoder1 > 0
        else text_encoder1
    )
    te2_ddp: nn.Module = (
        DDP(text_encoder2, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if args.lr_text_encoder2 > 0
        else text_encoder2
    )

    mu = 0.8
    sigma = 0.8

    scaler = GradScaler(enabled=mixed_precision)
    unet_dtype = next(unet.parameters()).dtype
    clip_params = [p for p in unet.parameters() if p.requires_grad]
    if args.lr_text_encoder1 > 0:
        clip_params += [p for p in text_encoder1.parameters() if p.requires_grad]
    if args.lr_text_encoder2 > 0:
        clip_params += [p for p in text_encoder2.parameters() if p.requires_grad]

    save_dir = os.path.dirname(args.save_path)
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_name = os.path.splitext(os.path.basename(args.save_path))[0]
    writer: Optional[SummaryWriter] = None
    if is_main_process:
        writer = SummaryWriter(log_dir=f"logs/{timestamp_str}_{session_name}")

    global_step = 0
    step = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(args.steps), disable=not is_main_process)
    accum_loss = 0.0
    accum_lpips_loss = 0.0

    if args.sample_every > 0 and args.sample_prompts and is_main_process:
        print("Generating initial sample images before training...")
        sample_and_save_images(
            0,
            unwrap_ddp(unet_ddp),
            unwrap_ddp(te1_ddp),
            unwrap_ddp(te2_ddp),
            args.sample_prompts,
            args.sample_seed,
            save_dir,
            tokenizer1,
            tokenizer2,
            autocast_dtype,
        )

    if args.initial_steps > 0:
        if is_main_process:
            print(f"Skipping to initial step {args.initial_steps}...")
        is_skipping.value = True

    try:
        while step < args.steps:
            sampler.set_epoch(epoch)
            if is_main_process:
                print(f"Starting epoch {epoch+1}...")
            current_epoch.value = epoch + 1

            epoch_losses = []
            epoch += 1

            for data in data_loader:
                if step < args.initial_steps:
                    step += 1
                    if step >= args.steps:
                        break
                    if pbar is not None:
                        pbar.update(1)
                    if step >= args.initial_steps - 20 and is_skipping.value:
                        is_skipping.value = False
                        if is_main_process:
                            print("Resuming training now.")
                    continue

                images = data[0].squeeze(0)
                input_ids1 = data[1].squeeze(0)
                input_ids2 = data[2].squeeze(0)
                original_sizes = data[3].squeeze(0)
                crop_sizes = data[4].squeeze(0)
                target_sizes = data[5].squeeze(0)
                assert (
                    original_sizes[0][0].item() > 0 and original_sizes[0][1].item() > 0
                ), "Illegal original size 0, maybe skipped all data in this epoch."

                images = images.to(device=device, dtype=unet_dtype, non_blocking=True)
                input_ids1 = input_ids1.to(device, non_blocking=True)
                input_ids2 = input_ids2.to(device, non_blocking=True)
                original_sizes = original_sizes.to(device, non_blocking=True)
                crop_sizes = crop_sizes.to(device, non_blocking=True)
                target_sizes = target_sizes.to(device, non_blocking=True)

                with autocast(device_type="cuda", dtype=autocast_dtype, enabled=mixed_precision):
                    context, y = prepare_conditioning(
                        input_ids1,
                        input_ids2,
                        original_sizes,
                        crop_sizes,
                        target_sizes,
                        te1_ddp,
                        te2_ddp,
                        dtype=unet_dtype,
                        device=device,
                    )

                B = images.shape[0]
                x0 = images

                if args.uniform_sampling:
                    t = sample_t_uniform(B, device=device)
                else:
                    t = sample_t(B, device=device, mu=mu, sigma=sigma)

                x_t, _ = add_noise_x0(x0, t, noise_scale=1.0)

                with autocast(device_type="cuda", dtype=autocast_dtype, enabled=mixed_precision):
                    x0_pred = unet_ddp(
                        x_t=x_t,
                        timesteps=t * (NUM_TRAIN_TIMESTEPS - 1),
                        context=context,
                        y=y,
                    )

                    if args.velocity_loss:
                        base_loss = velocity_loss_x_pred(x0, x_t, x0_pred, t)
                    else:
                        base_loss = torch.nn.functional.mse_loss(x0_pred, x0)

                lpips_loss_val = None
                if lpips_model is not None:
                    # モデル出力およびターゲット画像が1024x1024で大きいので、半分にリサイズしてLPIPS計算
                    x0_pred_resized = F.interpolate(x0_pred, scale_factor=0.5, mode="bilinear", align_corners=False)
                    x0_resized = F.interpolate(x0, scale_factor=0.5, mode="bilinear", align_corners=False)

                    # [-1,1] で LPIPS を計算（出力をクランプ）
                    lpips_pred = torch.clamp(x0_pred_resized, -1.0, 1.0).float()
                    lpips_target = torch.clamp(x0_resized, -1.0, 1.0).float()

                    lpips_loss_val = lpips_model(lpips_pred, lpips_target).mean()

                total_loss = base_loss
                if lpips_loss_val is not None:
                    total_loss = total_loss + args.lpips_lambda * lpips_loss_val

                total_loss_detached = reduce_tensor(total_loss.detach())
                accum_loss += total_loss_detached.item()
                if lpips_loss_val is not None:
                    lpips_detached = reduce_tensor(lpips_loss_val.detach())
                    accum_lpips_loss += lpips_detached.item()

                loss = total_loss / args.grad_accum_steps

                if mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_losses.append(total_loss_detached.item())

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

                    if is_main_process and writer is not None:
                        avg_accum_loss = accum_loss / args.grad_accum_steps
                        pbar.set_description(f"loss={avg_accum_loss:.4f}")
                        writer.add_scalar("train/loss_step", avg_accum_loss, global_step)
                        if lpips_loss_val is not None:
                            avg_accum_loss_lpips = accum_lpips_loss / args.grad_accum_steps
                            writer.add_scalar("train/lpips_step", avg_accum_loss_lpips, global_step)

                    accum_loss = 0.0
                    accum_lpips_loss = 0.0

                if is_main_process and args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    save_path_body, split_ext = os.path.splitext(args.save_path)
                    save_path = f"{save_path_body}_step{step + 1}{split_ext}"
                    save_state_to_checkpoint(
                        save_path,
                        unwrap_ddp(unet_ddp),
                        unwrap_ddp(te1_ddp),
                        unwrap_ddp(te2_ddp),
                        optimizer,
                        logit_scale,
                    )
                    print(f"Checkpoint saved at step {step + 1}: {save_path}")

                if is_main_process and args.sample_every > 0 and (step + 1) % args.sample_every == 0 and args.sample_prompts:
                    print(f"Generating sample images at step {step + 1}...")
                    sample_and_save_images(
                        step + 1,
                        unwrap_ddp(unet_ddp),
                        unwrap_ddp(te1_ddp),
                        unwrap_ddp(te2_ddp),
                        args.sample_prompts,
                        args.sample_seed,
                        save_dir,
                        tokenizer1,
                        tokenizer2,
                        autocast_dtype,
                    )

                step += 1
                if interrupt_event.is_set():
                    raise KeyboardInterrupt
                if pbar is not None:
                    pbar.update(1)

                if step >= args.steps:
                    break

            if is_main_process and writer is not None and epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch)
                print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

    except KeyboardInterrupt:
        if is_main_process:
            print("Training interrupted. Saving current checkpoint...")
            save_path_body, split_ext = os.path.splitext(args.save_path)
            interrupted_save_path = f"{save_path_body}_interrupted_step{step}{split_ext}"
            save_state_to_checkpoint(
                interrupted_save_path,
                unwrap_ddp(unet_ddp),
                unwrap_ddp(te1_ddp),
                unwrap_ddp(te2_ddp),
                optimizer,
                logit_scale,
            )
            print(f"Checkpoint saved: {interrupted_save_path}")

        if writer is not None:
            writer.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    if is_main_process:
        print("Saving final checkpoint...")
        save_state_to_checkpoint(
            args.save_path, unwrap_ddp(unet_ddp), unwrap_ddp(te1_ddp), unwrap_ddp(te2_ddp), optimizer, logit_scale
        )
        print(f"Training finished. Saved to {args.save_path}")

    if writer is not None:
        writer.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
