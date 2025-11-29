from datetime import datetime
from multiprocessing import Value
import os
from typing import Any, Optional
import torch
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import safetensors.torch

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
        state_dict: dict[str, Any],
        base_resolution: int,
        encoder_decoder_architecture: str,
        device: torch.device,
        current_epoch: Any,
        metadata_files: str,
        batch_size: int,
        tokenizer1,
        tokenizer2,
        seed: Optional[int] = None,
        is_skipping: Optional[Any] = None,
    ):
        super().__init__(current_epoch, metadata_files, batch_size, tokenizer1, tokenizer2, seed, is_skipping)
        self.state_dict = state_dict
        self.base_resolution = base_resolution
        self.encoder_decoder_architecture = encoder_decoder_architecture
        self.text_encoder1 = None
        self.text_encoder2 = None
        self.device = device

    def __getitem__(self, index):
        images, input_ids1, input_ids2, original_sizes, crop_sizes, target_sizes = super().__getitem__(index)
        if self.is_skipping is not None and self.is_skipping.value:
            return images, torch.empty((1, 77, 1)), torch.empty((1, 77, 1)), original_sizes, crop_sizes, target_sizes

        if self.text_encoder1 is None or self.text_encoder2 is None:
            # lazy load text encoders to avoid unnecessary GPU memory usage
            print("Loading text encoders into dataset...")
            self.text_encoder1, self.text_encoder2, _, _ = load_models_from_state_dict(
                self.state_dict,
                base_resolution=self.base_resolution,
                encoder_decoder_architecture=self.encoder_decoder_architecture,
            )
            self.text_encoder1.to(device=self.device)
            self.text_encoder2.to(device=self.device)
            self.text_encoder1.eval()
            self.text_encoder2.eval()
            self.state_dict = None  # free memory

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

    # make a copy to avoid modifying the original during loading
    model_state_dict_copy = dict(state_dict["model"]) if "model" in state_dict else dict(state_dict)

    text_encoder1, text_encoder2, unet, logit_scale = load_models_from_state_dict(
        state_dict, base_resolution=args.base_resolution, encoder_decoder_architecture=args.encoder_decoder_architecture
    )

    unet.to(device_main)
    print("Models loaded. Casting U-Net to float32 for stable training...")
    unet.to(torch.float32)  # ensure float32 for stable training even with mixed precision

    text_encoder1.eval()  # run on CPU
    text_encoder2.eval()  # run on CPU
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

    # DataSet側、つまりDataLoaderのworkerプロセスでテキストエンコーダをロードするためにstate_dictを渡す（無理やりすぎる～(;^ω^)
    dataset = EncodingImageDataset(
        model_state_dict_copy,
        args.base_resolution,
        args.encoder_decoder_architecture,
        device_secondary,
        current_epoch,
        args.metadata_files,
        args.batch_size,
        tokenizer1,
        tokenizer2,
        seed=args.seed,
        is_skipping=is_skipping,
    )
    del model_state_dict_copy, state_dict  # free memory

    # dataloader with batch_size=1 since ImageDataset already returns a batch
    print("Preparing data loader... Note: num_workers is set to 1 because text encoders are used in dataset.")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)

    mu = 0.8
    sigma = 0.8
    # t_eps = 5e-2

    scaler = GradScaler(enabled=mixed_precision)
    unet_dtype = next(unet.parameters()).dtype  # これいつもfloat32のはず
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
    accum_loss = torch.tensor(0.0, device=device_main)  # gradient accumulation中のloss合計
    accum_lpips_loss = torch.tensor(0.0, device=device_main)  # gradient accumulation中のLPIPS loss合計

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

                # このassertionは、original_sizesがCPU上にあるのでCUDAと同期しない
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

                    # デバッグ用: 勾配追跡を確認
                    # if lpips_model is not None:
                    #     x0_pred.retain_grad()

                    if args.velocity_loss:
                        # 4-a. v-loss 計算
                        base_loss = velocity_loss_x_pred(x0, x_t, x0_pred, t)
                    else:
                        # 4-b. x-pred loss 計算
                        base_loss = F.mse_loss(x0_pred, x0)

                # LPIPS計算 (PyTorchのAutogradに任せる修正版)
                lpips_loss_val = None
                lpips_loss_tensor = None

                if lpips_model is not None:
                    # assert x0_pred.requires_grad, "x0_pred must require grad for LPIPS loss computation."

                    # 計算グラフを切らずに、テンソルをsecondaryデバイスへ移動
                    # float32へキャストしてから移動（LPIPSの安定性のため）
                    x0_secondary = x0.detach().to(device_secondary, dtype=torch.float32)  # 勾配不要
                    x0_pred_secondary = x0_pred.to(device_secondary, dtype=torch.float32)
                    t_secondary = t.detach().to(device_secondary)  # 勾配不要

                    # secondaryデバイス上でリサイズとクランプ
                    # interpolateもautogradの対象になります
                    x0_pred_resized = F.interpolate(x0_pred_secondary, scale_factor=0.5, mode="bilinear", align_corners=False)
                    x0_resized = F.interpolate(x0_secondary, scale_factor=0.5, mode="bilinear", align_corners=False)

                    lpips_pred_clamped = torch.clamp(x0_pred_resized, -1.0, 1.0)
                    lpips_target_clamped = torch.clamp(x0_resized, -1.0, 1.0)

                    # LPIPS forward
                    lpips_out = lpips_model(lpips_pred_clamped, lpips_target_clamped)

                    # 重み付け
                    lpips_out = lpips_out * (1 - t_secondary).view(B, 1, 1, 1)
                    lpips_loss_tensor = lpips_out.mean()

                    # # デバッグ用: 勾配が流れているか確認
                    # lpips_loss_tensor.backward(retain_graph=True)  # テスト時のみ有効化
                    # assert x0_pred.grad is not None, "Gradient did not flow back to x0_pred"

                # Lossの合算とBackward
                # base_lossはmainデバイス、lpips_loss_tensorはsecondaryデバイスにありますが、
                # 加算時にPyTorchが自動的に処理、あるいは明示的に戻して加算します。

                total_loss = base_loss
                if lpips_loss_tensor is not None:
                    lpips_loss_tensor = lpips_loss_tensor.to(device_main)
                    lpips_loss_val = lpips_loss_tensor.detach()
                    total_loss = total_loss + args.lpips_lambda * lpips_loss_tensor

                total_loss = total_loss / args.grad_accum_steps
                if mixed_precision:
                    scaler.scale(total_loss).backward()  # 単にscaleするだけなのでLPIPS lossも含んでも良い
                else:
                    total_loss.backward()

                # ログ集計用の値計算
                base_loss_val = base_loss.detach()
                total_loss_val = base_loss_val
                if lpips_loss_val is not None:
                    total_loss_val = total_loss_val + lpips_loss_val * args.lpips_lambda

                # gradient accumulation中のlossを蓄積
                accum_loss += total_loss_val
                if lpips_loss_val is not None:
                    accum_lpips_loss += lpips_loss_val

                # epoch平均loss計算用に保存
                epoch_losses.append(total_loss_val * args.grad_accum_steps)  # tensor

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
                    accum_loss = accum_loss.item()
                    accum_lpips_loss = accum_lpips_loss.item()
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
                epoch_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in epoch_losses]
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
