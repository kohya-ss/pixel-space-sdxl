import torch
import safetensors
from accelerate import init_empty_weights
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

import pixel_sdxl.sdxl_pixel_unet as sdxl_pixel_unet


def convert_sdxl_text_encoder_2_checkpoint(checkpoint):
    SDXL_KEY_PREFIX = "conditioner.embedders.1.model."

    # SD2のと、基本的には同じ。logit_scaleを後で使うので、それを追加で返す
    # logit_scaleはcheckpointの保存時に使用する
    def convert_key(key):
        # common conversion
        key = key.replace(SDXL_KEY_PREFIX + "transformer.", "text_model.encoder.")
        key = key.replace(SDXL_KEY_PREFIX, "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = key.replace("text_model.text_projection", "text_projection.weight")
        elif ".logit_scale" in key:
            key = None  # 後で処理する
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        # ckpt from comfy has this key: text_model.encoder.text_model.embeddings.position_ids
        elif ".embeddings.position_ids" in key:
            key = None  # remove this key: position_ids is not used in newer transformers
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace(SDXL_KEY_PREFIX + "transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # logit_scale はDiffusersには含まれないが、保存時に戻したいので別途返す
    logit_scale = checkpoint.get(SDXL_KEY_PREFIX + "logit_scale", None)

    # temporary workaround for text_projection.weight.weight for Playground-v2
    if "text_projection.weight.weight" in new_sd:
        print("convert_sdxl_text_encoder_2_checkpoint: convert text_projection.weight.weight to text_projection.weight")
        new_sd["text_projection.weight"] = new_sd["text_projection.weight.weight"]
        del new_sd["text_projection.weight.weight"]

    return new_sd, logit_scale


def convert_text_encoder_2_state_dict_to_sdxl(checkpoint, logit_scale):
    def convert_key(key):
        # position_idsの除去
        if ".position_ids" in key:
            return None

        # common
        key = key.replace("text_model.encoder.", "transformer.")
        key = key.replace("text_model.", "")
        if "layers" in key:
            # resblocks conversion
            key = key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in key:
                key = key.replace(".layer_norm", ".ln_")
            elif ".mlp." in key:
                key = key.replace(".fc1.", ".c_fc.")
                key = key.replace(".fc2.", ".c_proj.")
            elif ".self_attn.out_proj" in key:
                key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif ".self_attn." in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {key}")
        elif ".position_embedding" in key:
            key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif ".token_embedding" in key:
            key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif "text_projection" in key:  # no dot in key
            key = key.replace("text_projection.weight", "text_projection")
        elif "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ln_final")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if "layers" in key and "q_proj" in key:
            # 三つを結合
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    if logit_scale is not None:
        new_sd["logit_scale"] = logit_scale

    return new_sd


def load_models_from_state_dict(
    state_dict: dict[str, torch.Tensor], base_resolution: int = 64, encoder_decoder_architecture: str = "default"
) -> tuple[CLIPTextModel, CLIPTextModelWithProjection, sdxl_pixel_unet.SDXLPixelUNet, torch.Tensor | None]:
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Check and convert keys if necessary
    if "model.diffusion_model.input_blocks.0.0.weight" in state_dict:
        # original SDXL checkpoint
        print("Prepare new U-Net state dict...")
        unet = sdxl_pixel_unet.SDXLPixelUNet(
            base_resolution=base_resolution, encoder_decoder_architecture=encoder_decoder_architecture
        )
        pixel_unet_sd = unet.state_dict()

        print("Loading original SDXL checkpoint...")

        # Remove layers not used in pixel-space U-Net
        num_removed = 0
        for key in list(state_dict.keys()):
            if key.startswith("model.diffusion_model."):
                unet_key = key.replace("model.diffusion_model.", "")
                if unet_key not in pixel_unet_sd:
                    # print(f"Removing unused layer from state dict: {key}")
                    state_dict.pop(key)  # remove unused layer
                    num_removed += 1
        print(f"Removed {num_removed} unused layers from state dict. Remaining layers: {len(state_dict)}")

        # Add layers missing in pixel-space U-Net
        num_added = 0
        for key in pixel_unet_sd:
            state_dict_key = "model.diffusion_model." + key
            if state_dict_key not in state_dict:
                state_dict[state_dict_key] = pixel_unet_sd[key]  # add missing layer, initialized by pixel-space U-Net default init
                num_added += 1
        print(f"Added {num_added} missing layers to state dict. Total layers: {len(state_dict)}")

    # U-Net
    print("building U-Net")
    with init_empty_weights():
        unet = sdxl_pixel_unet.SDXLPixelUNet(base_resolution=base_resolution)

    print("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)
    info = unet.load_state_dict(unet_sd, strict=True, assign=True)
    print(f"U-Net: {info}")

    # Text Encoders
    print("building text encoders")

    # Text Encoder 1 is same to Stability AI's SDXL
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        text_model1 = CLIPTextModel._from_config(text_model1_cfg)  # noqa: F821

    # Text Encoder 2 is different from Stability AI's SDXL. SDXL uses open clip, but we use the model from HuggingFace.
    # Note: Tokenizer from HuggingFace is different from SDXL. We must use open clip's tokenizer.
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    print("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)

    # 最新の transformers では position_ids を含むとエラーになるので削除 / remove position_ids for latest transformers
    if "text_model.embeddings.position_ids" in te1_sd:
        te1_sd.pop("text_model.embeddings.position_ids")

    info1 = text_model1.load_state_dict(te1_sd, strict=True, assign=True)
    print(f"text encoder 1: {info1}")

    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd)
    info2 = text_model2.load_state_dict(converted_sd, strict=True, assign=True)
    print(f"text encoder 2: {info2}")

    return text_model1, text_model2, unet, logit_scale


def serialize_models_to_state_dict(
    text_model1: CLIPTextModel,
    text_model2: CLIPTextModelWithProjection,
    unet: sdxl_pixel_unet.SDXLPixelUNet,
    logit_scale: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    print("Serializing models to state dict...")
    state_dict = {}

    # U-Net
    unet_sd = unet.state_dict()
    for key in unet_sd:
        state_dict["model.diffusion_model." + key] = unet_sd[key]

    # Text Encoder 1
    te1_sd = text_model1.state_dict()
    for key in te1_sd:
        state_dict["conditioner.embedders.0.transformer." + key] = te1_sd[key]

    # Text Encoder 2
    te2_sd = convert_text_encoder_2_state_dict_to_sdxl(text_model2.state_dict(), logit_scale)
    for key in te2_sd:
        state_dict["conditioner.embedders.1.model." + key] = te2_sd[key]
    if logit_scale is not None:
        state_dict["conditioner.embedders.1.model.logit_scale"] = logit_scale

    return state_dict
