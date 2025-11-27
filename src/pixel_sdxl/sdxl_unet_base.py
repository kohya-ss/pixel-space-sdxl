import math
from typing import Optional
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange


# IN_CHANNELS: int = 4
# OUT_CHANNELS: int = 4
ADM_IN_CHANNELS: int = 2816
CONTEXT_DIM: int = 2048
MODEL_CHANNELS: int = 320
TIME_EMBED_DIM = 320 * 4

USE_REENTRANT = True
USE_XFORMERS = True
USE_FLASH_ATTENTION = False


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings: flipped from Diffusers original ver because always flip_sin_to_cos=True
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        if self.weight.dtype != torch.float32:
            return super().forward(x)
        return super().forward(x.float()).type(x.dtype)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(TIME_EMBED_DIM, out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Identity(),  # to make state_dict compatible with original model
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

        self.gradient_checkpointing = False

    def forward_body(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h

    def forward(self, x, emb):
        if self.training and self.gradient_checkpointing:
            # print("ResnetBlock2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, emb, use_reentrant=USE_REENTRANT)
        else:
            x = self.forward_body(x, emb)

        return x


class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)

        return hidden_states

    def forward(self, hidden_states):
        if self.training and self.gradient_checkpointing:
            # print("Downsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, use_reentrant=USE_REENTRANT
            )
        else:
            hidden_states = self.forward_body(hidden_states)

        return hidden_states


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        # no dropout here

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward_memory_efficient_xformers(self, x, context=None, mask=None):
        import xformers.ops

        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる
        del q, k, v

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward_flash_attention(self, x, context=None, mask=None):
        from flash_attn.flash_attn_interface import flash_attn_func

        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        del q, k, v

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward_sdpa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward(self, x, context=None, mask=None):
        if USE_XFORMERS:
            return self.forward_memory_efficient_xformers(x, context, mask)
        else:
            return self.forward_sdpa(x, context, mask)


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        inner_dim = int(dim * 4)  # mult is always 4

        self.net = nn.ModuleList([])
        # project in
        self.net.append(GEGLU(dim, inner_dim))
        # project dropout
        self.net.append(nn.Identity())  # nn.Dropout(0)) # dummy for dropout with 0
        # project out
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int, upcast_attention: bool = False
    ):
        super().__init__()

        self.gradient_checkpointing = False

        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )
        self.ff = FeedForward(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)

    def forward_body(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

    def forward(self, hidden_states, context=None, timestep=None):
        if self.training and self.gradient_checkpointing:
            # print("BasicTransformerBlock: checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, context, timestep, use_reentrant=USE_REENTRANT
            )
        else:
            output = self.forward_body(hidden_states, context, timestep)

        return output


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # self.norm = GroupNorm32(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        blocks = []
        for _ in range(num_transformer_layers):
            blocks.append(
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            )

        self.transformer_blocks = nn.ModuleList(blocks)

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.gradient_checkpointing = False

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None):
        # 1. Input
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        return output


class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        # -> This issue is fixed in PyTorch 2.1.0
        # dtype = hidden_states.dtype
        # if dtype == torch.bfloat16:
        #     hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        # if dtype == torch.bfloat16:
        #     hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states

    def forward(self, hidden_states, output_size=None):
        if self.training and self.gradient_checkpointing:
            # print("Upsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, output_size, use_reentrant=USE_REENTRANT
            )
        else:
            hidden_states = self.forward_body(hidden_states, output_size)

        return hidden_states


def add_sdxl_unet_modules(self, base_resolution: int = 64):
    self.model_channels = MODEL_CHANNELS
    self.time_embed_dim = TIME_EMBED_DIM
    self.adm_in_channels = ADM_IN_CHANNELS

    # time embedding
    self.time_embed = nn.Sequential(
        nn.Linear(self.model_channels, self.time_embed_dim),
        nn.SiLU(),
        nn.Linear(self.time_embed_dim, self.time_embed_dim),
    )

    # label embedding
    self.label_emb = nn.Sequential(
        nn.Sequential(
            nn.Linear(self.adm_in_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
    )

    self.input_blocks = nn.ModuleList([])

    # No first conv: 4 channels -> model_channels
    self.input_blocks.append(
        nn.ModuleList([nn.Identity()])
    )  # dummy for first conv to make state_dict compatible with original model

    # No residual blocks before downsampling
    self.input_blocks.append(nn.ModuleList([nn.Identity()]))
    self.input_blocks.append(nn.ModuleList([nn.Identity()]))

    # No first downsample: 128x128 -> 64x64
    self.input_blocks.append(nn.ModuleList([nn.Identity()]))

    # level 1: 64x64.
    if base_resolution == 64:
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(1 if i == 0 else 2) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        # downsample to level 2: 64x64 -> 32x32
        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=2 * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
            )
        )
    else:
        self.input_blocks.append(nn.ModuleList([nn.Identity()]))
        self.input_blocks.append(nn.ModuleList([nn.Identity()]))
        self.input_blocks.append(nn.ModuleList([nn.Identity()]))

    # level 2
    for i in range(2):
        layers = [
            ResnetBlock2D(
                in_channels=(2 if i == 0 else 4) * self.model_channels,
                out_channels=4 * self.model_channels,
            ),
            Transformer2DModel(
                num_attention_heads=4 * self.model_channels // 64,
                attention_head_dim=64,
                in_channels=4 * self.model_channels,
                num_transformer_layers=10,
                use_linear_projection=True,
                cross_attention_dim=2048,
            ),
        ]
        self.input_blocks.append(nn.ModuleList(layers))

    # mid
    self.middle_block = nn.ModuleList(
        [
            ResnetBlock2D(
                in_channels=4 * self.model_channels,
                out_channels=4 * self.model_channels,
            ),
            Transformer2DModel(
                num_attention_heads=4 * self.model_channels // 64,
                attention_head_dim=64,
                in_channels=4 * self.model_channels,
                num_transformer_layers=10,
                use_linear_projection=True,
                cross_attention_dim=2048,
            ),
            ResnetBlock2D(
                in_channels=4 * self.model_channels,
                out_channels=4 * self.model_channels,
            ),
        ]
    )

    # output
    self.output_blocks = nn.ModuleList([])

    # level 2
    for i in range(3):
        layers = [
            ResnetBlock2D(
                in_channels=4 * self.model_channels + (4 if i <= 1 else 2) * self.model_channels,
                out_channels=4 * self.model_channels,
            ),
            Transformer2DModel(
                num_attention_heads=4 * self.model_channels // 64,
                attention_head_dim=64,
                in_channels=4 * self.model_channels,
                num_transformer_layers=10,
                use_linear_projection=True,
                cross_attention_dim=2048,
            ),
        ]
        if i == 2 and base_resolution == 64:
            layers.append(
                Upsample2D(
                    channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                )
            )

        self.output_blocks.append(nn.ModuleList(layers))

    if base_resolution == 64:
        # level 1
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=2 * self.model_channels + (4 if i == 0 else (2 if i == 1 else 1)) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            # Remove upsample at the last block

            self.output_blocks.append(nn.ModuleList(layers))

    # No level 0: No need to add Identity
