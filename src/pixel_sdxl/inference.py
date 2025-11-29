import argparse
from datetime import datetime
import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm
import safetensors.torch

from pixel_sdxl import train_utils
from pixel_sdxl.model_utils import load_models_from_state_dict
from pixel_sdxl.sdxl_pixel_unet import SDXLPixelUNet
from pixel_sdxl.text_encoder_utils import encode_tokens, get_sdxl_tokenizers, tokenize


NUM_TRAIN_TIMESTEPS = 1000
T_EPS = 0.05  # 1e-3


@dataclass
class InferenceConfig:
    prompt: str
    negative_prompt: str = ""
    height: int = 1024
    width: int = 1024
    num_steps: int = 30
    seed: int = 0
    schedule: Literal["linear", "flow_shift"] = "linear"
    cfg_scale: float = 7.0
    flow_shift: float = 3.0
    precision: Literal["float16", "float32", "bfloat16"] = "float16"


def make_timesteps(num_steps: int, schedule: str, flow_shift: float, device):
    """
    linear:     1.0 (ノイズ) → 0.0 (画像) に線形に下る時間ステップ列
    flow_shift: 1.0 → 0.0 を flow-based の変換でシフトした時間ステップ列
    return: [T+1] テンソル
    """
    if schedule == "flow_shift":
        sigmas = torch.linspace(1, 0, num_steps + 1)
        sigmas = (flow_shift * sigmas) / (1 + (flow_shift - 1) * sigmas)
        t_steps = sigmas.to(device=device)
    else:
        # 例: num_steps=50 → [1.0, 0.98, 0.96, ..., 0.0]
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return t_steps


def _prepare_conditioning(
    prompt: str,
    height: int,
    width: int,
    tokenizer1,
    tokenizer2,
    text_encoder1,
    text_encoder2,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids1, input_ids2 = tokenize(tokenizer1, tokenizer2, prompt)
    original_sizes = torch.tensor([[height, width]], device=device, dtype=torch.float32)
    crop_top_lefts = torch.tensor([[0.0, 0.0]], device=device, dtype=torch.float32)
    target_sizes = torch.tensor([[height, width]], device=device, dtype=torch.float32)

    hidden_states1, hidden_states2, pooled2 = encode_tokens(input_ids1, input_ids2, text_encoder1, text_encoder2)
    context = torch.cat([hidden_states1, hidden_states2], dim=-1).to(device=device, dtype=dtype)
    size_embeddings = train_utils.get_size_embeddings(original_sizes, crop_top_lefts, target_sizes, device=device).to(dtype)
    y = torch.cat([pooled2.to(device=device, dtype=dtype), size_embeddings], dim=-1)
    return context, y


@torch.no_grad()
def generate_image(
    unet: SDXLPixelUNet,
    text_encoder1,
    text_encoder2,
    tokenizer1,
    tokenizer2,
    config: InferenceConfig,
    autocast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    device = unet.device
    dtype = unet.dtype
    if config.height % unet.patch_size != 0 or config.width % unet.patch_size != 0:
        raise ValueError(f"height and width must be divisible by patch size ({unet.patch_size})")

    generator = torch.Generator(device=device).manual_seed(config.seed)

    with autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        context, y = _prepare_conditioning(
            prompt=config.prompt,
            height=config.height,
            width=config.width,
            tokenizer1=tokenizer1,
            tokenizer2=tokenizer2,
            text_encoder1=text_encoder1,
            text_encoder2=text_encoder2,
            device=text_encoder1.device,
            dtype=text_encoder1.dtype,
        )
        uncond_context, uncond_y = _prepare_conditioning(
            prompt=config.negative_prompt,
            height=config.height,
            width=config.width,
            tokenizer1=tokenizer1,
            tokenizer2=tokenizer2,
            text_encoder1=text_encoder1,
            text_encoder2=text_encoder2,
            device=text_encoder1.device,
            dtype=text_encoder1.dtype,
        )
    context = context.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)
    uncond_context = uncond_context.to(device=device, dtype=dtype)
    uncond_y = uncond_y.to(device=device, dtype=dtype)

    # 1. 時間ステップ列を作る
    t_steps = make_timesteps(config.num_steps, config.schedule, config.flow_shift, device=device)  # [T+1]
    print(f"Using {config.schedule} timestep schedule: {t_steps}")

    # 2. 初期状態 x_{t0=1}: 純ノイズ
    batch_size = 1
    x = torch.randn(batch_size, 3, config.height, config.width, device=device, dtype=dtype, generator=generator)

    use_autocast = autocast_dtype is not None
    t_clip = T_EPS  # t が 0 に近づきすぎないようにクリップ

    # 3. ループ: t_i → t_{i+1} （1→0へ降りていく）
    for i in tqdm(range(config.num_steps)):
        t = t_steps[i]  # スカラー
        s = t_steps[i + 1]  # 次の時刻 (s < t)
        t_vec = torch.full((batch_size, 1, 1, 1), t, device=device, dtype=dtype)
        t_vec = torch.clamp(t_vec, min=t_clip)
        s_vec = torch.full((batch_size, 1, 1, 1), s, device=device, dtype=dtype)

        model_t = (t * (NUM_TRAIN_TIMESTEPS - 1)).reshape(1)
        # print(f"Step {i+1}/{config.num_steps}: t={t:.4f} -> s={s:.4f}, model_t={model_t.item():.1f}")

        with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            # 3-1. モデルで x0 を予測 (x-pred)
            x_cond = unet(x_t=x, timesteps=model_t, context=context, y=y)
            x_uncond = unet(x_t=x, timesteps=model_t, context=uncond_context, y=uncond_y)

        # # 3-2. 予測された x0 からvelocity を計算
        # v_cond = (x_cond - x) / t_vec
        # v_uncond = (x_uncond - x) / t_vec
        # # classifier-free guidance
        # v_pred = v_uncond + config.cfg_scale * (v_cond - v_uncond)
        # # 4. 次のステップ s へ
        # x_next = x + (s_vec - t_vec) * v_pred
        # x = x_next

        # 3-2. 予測された x0 から次のステップ s への x を計算
        # classifier-free guidance
        x0_pred = x_uncond + config.cfg_scale * (x_cond - x_uncond)

        # 4. 次のステップ s へ
        eps_pred = (x - (1 - t_vec) * x0_pred) / t_vec
        x_next = (1 - s_vec) * x0_pred + s_vec * eps_pred
        x = x_next

    # ループ終了時: x ≈ x0（t=0）
    return x  # [B,3,H,W]


def tensor_to_pil_image(x: torch.Tensor) -> Image.Image:
    x = x.squeeze(0).detach().cpu()
    x = x.clamp(-1, 1)
    x = (x + 1) * 127.5
    x = x.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Pixel-space SDXL inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to a safetensors checkpoint.")
    parser.add_argument("--base_resolution", type=int, default=64, help="Base resolution for U-Net.")
    parser.add_argument(
        "--encoder_decoder_architecture",
        type=str,
        choices=["default", "conv"],
        default="default",
        help="Encoder-decoder architecture type.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt text.")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated images.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height (must be divisible by 16).")
    parser.add_argument("--width", type=int, default=1024, help="Output image width (must be divisible by 16).")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--timestep_schedule",
        choices=["linear", "flow_shift"],
        default="linear",
        help="Schedule for timesteps: linear or logit-normal shifted (flow_shift).",
    )
    parser.add_argument("--flow_shift", type=float, default=3.0, help="Mean used in flow_shift schedule.")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--precision",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Computation dtype for models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer1, tokenizer2 = get_sdxl_tokenizers()

    print("Loading models from checkpoint...")
    if args.checkpoint.endswith(".ckpt") or args.checkpoint.endswith(".pt"):
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    else:
        state_dict = safetensors.torch.load(open(args.checkpoint, "rb").read())
    text_encoder1, text_encoder2, unet, _ = load_models_from_state_dict(
        state_dict, base_resolution=args.base_resolution, encoder_decoder_architecture=args.encoder_decoder_architecture
    )

    config = InferenceConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_steps=args.num_inference_steps,
        seed=args.seed,
        schedule=args.timestep_schedule,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        precision=args.precision,
    )

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[config.precision]

    unet = unet.to(device, dtype=dtype).eval()
    text_encoder1 = text_encoder1.to(device, dtype=dtype).eval()
    text_encoder2 = text_encoder2.to(device, dtype=dtype).eval()

    image_tensor = generate_image(
        unet=unet,
        text_encoder1=text_encoder1,
        text_encoder2=text_encoder2,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        config=config,
        autocast_dtype=dtype if dtype in (torch.float16, torch.bfloat16) else None,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = datetime.now().strftime("generated_%Y%m%d_%H%M%S") + f"_{args.seed}.png"
    output_path = os.path.join(args.output_dir, output_name)
    tensor_to_pil_image(image_tensor).save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
