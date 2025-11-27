from typing import Any, Optional
import torch
from torch import nn

import pixel_sdxl.sdxl_unet_base as sdxl_unet_base


def get_parameter_dtype(parameter: torch.nn.Module):
    return next(parameter.parameters()).dtype


def get_parameter_device(parameter: torch.nn.Module):
    return next(parameter.parameters()).device


class Patchify(nn.Module):
    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, C * P * P, H//P, W//P]
        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "H,W must be divisible by patch_size"

        # [B, C, H//P, P, W//P, P]
        x = x.view(B, C, H // P, P, W // P, P)
        # [B, C, P, P, H//P, W//P]
        x = x.permute(0, 1, 3, 5, 2, 4)
        # [B, C * P * P, H//P, W//P]
        x = x.reshape(B, C * P * P, H // P, W // P)
        return x


class Unpatchify(nn.Module):
    def __init__(self, patch_size: int = 16, out_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C*P*P, H', W']  (C=out_channels)
        return: [B, C, H'*P, W'*P]
        """
        B, CP2, H_, W_ = x.shape
        P = self.patch_size
        C = self.out_channels
        assert CP2 == C * P * P

        # [B, C, P, P, H', W']
        x = x.view(B, C, P, P, H_, W_)
        # [B, C, H', P, W', P]
        x = x.permute(0, 1, 4, 2, 5, 3)
        # [B, C, H'*P, W'*P]
        x = x.reshape(B, C, H_ * P, W_ * P)
        return x


# 実験中に使ったパッチエンコーダ・デコーダ（64x64用）　現在は未使用


class PatchEncoder64(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,  # 3 * 16 * 16
        mid_channels: int = 320,  # SDXL model channel
        hidden_channels: int = 1024,
        num_blocks: int = 2,
    ):
        super().__init__()
        layers = []

        # 1x1 convで 768 → mid_channels に圧縮
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1))
        layers.append(nn.GroupNorm(32, mid_channels))
        layers.append(nn.SiLU())

        # いくつか residual conv block
        for _ in range(num_blocks):
            layers.append(ResidualConvBlock(mid_channels, hidden_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 768, 64, 64] → [B, 320, 64, 64]
        return self.net(x)


class PatchDecoder64(nn.Module):
    def __init__(
        self,
        in_channels: int = 640,
        hidden_channels: int = 1024,
        out_channels: int = 768,  # 3 * 16 * 16
        num_blocks: int = 2,
    ):
        super().__init__()
        layers = []

        for _ in range(num_blocks):
            layers.append(ResidualConvBlock(in_channels, hidden_channels))

        # 最後に 1x1 conv で 640 → 768 に戻す
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 640, 64, 64] → [B, 768, 64, 64]
        return self.net(x)


class PatchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3072,  # 3 * 32 * 32
        hidden_channels: int = 512,
        out_channels: int = 640,  # SDXL model channel
        num_blocks: int = 4,
    ):
        super().__init__()
        layers = []

        # 1x1 convで 3072 → hidden_channels に圧縮
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
        layers.append(nn.GroupNorm(32, hidden_channels))
        layers.append(nn.SiLU())

        # いくつか residual conv block
        for _ in range(num_blocks):
            layers.append(ResidualConvBlock(hidden_channels, hidden_channels))

        # 最後に 1x1 conv で hidden → out_channels
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3072, 32, 32] → [B, 640, 32, 32]
        return self.net(x)


class PatchDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1280,
        hidden_channels: int = 512,
        out_channels: int = 3072,  # 3 * 32 * 32
        num_blocks: int = 4,
    ):
        super().__init__()
        layers = []

        # 最初に 1x1 conv で in_channels → hidden_channels
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
        layers.append(nn.GroupNorm(32, hidden_channels))
        layers.append(nn.SiLU())

        # いくつか residual conv block
        for _ in range(num_blocks):
            layers.append(ResidualConvBlock(hidden_channels, hidden_channels))

        # 最後に 1x1 conv で 512 → 3072 に戻す
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1280, 32, 32] → [B, 3072, 32, 32]
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h + residual)
        return h


class SDXLPixelUNet(nn.Module):
    def __init__(
        self,
        base_resolution: int = 64,
        prediction_type: str = "x",  # "x" or "epsilon" or "v"
    ):
        super().__init__()

        in_channels = 3
        if base_resolution == 64:
            patch_size = 16
            base_channels = 640  # 64x64のch
            patch_hidden = 512
            num_res_blocks = 3
        elif base_resolution == 32:
            patch_size = 32
            base_channels = 1280  # 32x32のch
            patch_hidden = 768
            num_res_blocks = 4

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.prediction_type = prediction_type

        # パッチ化モジュール
        self.patchify = Patchify(patch_size=patch_size)
        self.unpatchify = Unpatchify(patch_size=patch_size, out_channels=in_channels)

        # パッチ encoder / decoder
        if base_resolution == 64:
            self.patch_encoder = PatchEncoder(
                in_channels=in_channels * patch_size * patch_size,  # 3 * 16 * 16 = 768
                hidden_channels=patch_hidden,
                out_channels=base_channels // 2,  # 320
                num_blocks=num_res_blocks,
            )
            self.patch_decoder = PatchDecoder(
                in_channels=base_channels,  # 640
                hidden_channels=patch_hidden,
                out_channels=in_channels * patch_size * patch_size,  # 768
                num_blocks=num_res_blocks,
            )
        elif base_resolution == 32:
            self.patch_encoder = PatchEncoder(
                in_channels=in_channels * patch_size * patch_size,  # 3 * 32 * 32 = 3072
                hidden_channels=patch_hidden,
                out_channels=base_channels // 2,  # 640
                num_blocks=num_res_blocks,
            )
            self.patch_decoder = PatchDecoder(
                in_channels=base_channels,  # 1280
                hidden_channels=patch_hidden,
                out_channels=in_channels * patch_size * patch_size,  # 3072
                num_blocks=num_res_blocks,
            )

        # 32x32または64x64以降の SDXL U-Net 本体
        sdxl_unet_base.add_sdxl_unet_modules(self, base_resolution=base_resolution)

        self.gradient_checkpointing = False

    def is_unet_body_parameter(self, name: str) -> bool:
        # SDXL U-Net 本体のパラメータかどうかを判定する
        return (
            name.startswith("input_blocks")
            or name.startswith("middle_block")
            or name.startswith("output_blocks")
            or name.startswith("time_embed")
            or name.startswith("label_emb")
        )

    def is_omitted_parameter(self, name: str, base_resolution: int) -> bool:
        # SDXL U-Netの省略されたレイヤーのパラメータかどうかを判定する
        is_input_blocks = name.startswith("input_blocks")
        is_output_blocks = name.startswith("output_blocks")
        if is_input_blocks or is_output_blocks:
            block_idx = int(name.split(".")[1])
            if base_resolution == 64:
                if is_input_blocks and block_idx <= 3:
                    return True
                if is_output_blocks and block_idx >= 6:
                    return True
            elif base_resolution == 32:
                if is_input_blocks and block_idx <= 6:
                    return True
                if is_output_blocks and block_idx >= 3:
                    return True
        return False

    @property
    def dtype(self) -> torch.dtype:
        # `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        # `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        return get_parameter_device(self)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.set_gradient_checkpointing(True)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.set_gradient_checkpointing(False)

    def set_gradient_checkpointing(self, value=False):
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block.modules():
                if hasattr(module, "gradient_checkpointing"):
                    # print(f{module.__class__.__name__} {module.gradient_checkpointing} -> {value}")
                    module.gradient_checkpointing = value

    def call_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        パッチ化とエンコードのみを行うユーティリティ関数
        """
        x_p = self.patchify(x)
        h = self.patch_encoder(x_p)
        return h

    def call_decoder(self, h: torch.Tensor) -> torch.Tensor:
        """
        デコードとアンパッチ化のみを行うユーティリティ関数
        """
        x_p_hat = self.patch_decoder(h)
        x_hat = self.unpatchify(x_p_hat)
        return x_hat

    def forward(
        self,
        x_t: torch.Tensor,  # [B, 3, 1024, 1024] ノイズ付き画像（ピクセル）
        timesteps: torch.Tensor,  # [B] or []
        context: torch.Tensor,  # テキスト条件 (B, L, D)
        y: torch.Tensor,  # [B, 640] or [B, 320] ADM条件
        added_cond_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        戻り値:
          - prediction_type="x"   → x0_hat (ピクセル空間)
          - "epsilon"/"v"         → その予測（必要なら外で変換）
        """

        # 1) pixel → patchify
        # x_p: [B, 3*P*P=768, 64, 64]
        x_p = self.patchify(x_t)

        # 2) パッチ encoder → [B, 640, 64, 64]
        h = self.patch_encoder(x_p)

        # 3) SDXL U-Net
        hs = []
        t_emb = sdxl_unet_base.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x_t.dtype)
        emb = self.time_embed(t_emb)
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # print(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, sdxl_unet_base.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_unet_base.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        hs.append(h)  # 最初の入力もスキップ接続用に保存

        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = call_module(module, h, emb, context)

        # 4) パッチ decoder → [B, 768, 64, 64]
        x_p_hat = self.patch_decoder(h)

        # 5) unpatchify → [B, 3, 1024, 1024]
        x_hat = self.unpatchify(x_p_hat)

        # 6) 返す量 (prediction_type) に応じて変換
        if self.prediction_type == "x":
            # x-prediction: そのまま x0_hat を返す
            return x_hat

        elif self.prediction_type == "epsilon":
            # 必要ならここで xt, t から eps_hat を計算して返す
            # xt = x_t
            # alpha_t, sigma_t は scheduler から取得
            # eps_hat = (xt - alpha_t * x_hat) / sigma_t
            raise NotImplementedError("epsilon prediction conversion not implemented here")

        elif self.prediction_type == "v":
            # v = alpha_t * eps - sigma_t * x0 みたいな定義に従って計算
            raise NotImplementedError("v prediction conversion not implemented here")

        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")


if __name__ == "__main__":
    import time
    import sys
    import safetensors.torch
    from pixel_sdxl import model_utils

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt == "None":
        ckpt = None
    base_resolution = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    # --- test Patchify / Unpatchify
    # テスト
    B, C, H, W = 2, 3, 1024, 1024
    P = 32

    x_orig = torch.randn(B, C, H, W)

    patchify = Patchify(patch_size=P)
    unpatchify = Unpatchify(patch_size=P, out_channels=C)

    # Forward
    x_patched = patchify(x_orig)
    print(f"Patched shape: {x_patched.shape}")  # [2, 768, 64, 64]

    # Backward
    x_restored = unpatchify(x_patched)
    print(f"Restored shape: {x_restored.shape}")  # [2, 3, 1024, 1024]

    # 完全に復元できているか確認
    diff = (x_orig - x_restored).abs().max()
    print(f"Max difference: {diff.item()}")  # 0.0 になるはず

    if diff < 1e-6:
        print("✓ Patchify/Unpatchify の逆変換は正しく動作しています")
    else:
        print("✗ 逆変換が正しくありません！")

    print("create unet")
    if ckpt is not None:
        print(f"Testing loading from checkpoint: {ckpt}, base_resolution={base_resolution}")
        state_dict = (
            torch.load(ckpt, map_location="cpu")
            if not ckpt.endswith(".safetensors")
            else safetensors.torch.load(open(ckpt, "rb").read())
        )
        _, _, unet, _ = model_utils.load_models_from_state_dict(state_dict, base_resolution=base_resolution)
        unet.to(torch.float32)  # for mixed precision test
    else:
        unet = SDXLPixelUNet(base_resolution=base_resolution, prediction_type="x")

    unet.to("cuda")
    unet.enable_gradient_checkpointing()
    unet.train()

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"number of parameters: {n_params / 1e6:.2f} M")

    n_unet_params = sum(p.numel() for n, p in unet.named_parameters() if unet.is_unet_body_parameter(n))
    print(f"number of U-Net parameters: {n_unet_params / 1e6:.2f} M")
    print(f"number of non-U-Net parameters: {(n_params - n_unet_params) / 1e6:.2f} M")

    # 使用メモリ量確認用の疑似学習ループ
    print("preparing optimizer")

    # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

    # import bitsandbytes
    import bitsandbytes

    optimizer = bitsandbytes.adam.Adam8bit(unet.parameters(), lr=1e-3)  # not working
    # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
    # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

    import transformers

    # optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2

    scaler = torch.amp.GradScaler(enabled=True)

    print("start training")
    steps = 10
    batch_size = 1

    for step in range(steps):
        print(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 3, 1024, 1024).cuda()  # 1024x1024
        t = torch.randint(low=0, high=10, size=(batch_size,), device="cuda")
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, sdxl_unet_base.ADM_IN_CHANNELS).cuda()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = unet(x, t, ctx, y)
            print(output.dtype, output.shape)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    print(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")
