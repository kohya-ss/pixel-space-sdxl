import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


def get_sdxl_tokenizers() -> tuple[CLIPTokenizer, CLIPTokenizer]:
    tokenizer1 = CLIPTokenizer.from_pretrained(TOKENIZER1_PATH)
    tokenizer2 = CLIPTokenizer.from_pretrained(TOKENIZER2_PATH)
    tokenizer2.pad_token_id = 0  # use 0 as pad token for tokenizer2
    return tokenizer1, tokenizer2


def get_input_ids(tokenizer: CLIPTokenizer, text: str) -> torch.Tensor:
    max_length = tokenizer.model_max_length  # usually 77
    input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids[0]
    return input_ids


def tokenize(tokenizer1: CLIPTokenizer, tokenizer2: CLIPTokenizer, text: str | list[str]) -> list[torch.Tensor]:
    text = [text] if isinstance(text, str) else text
    return (
        torch.stack([get_input_ids(tokenizer1, t) for t in text], dim=0),
        torch.stack([get_input_ids(tokenizer2, t) for t in text], dim=0),
    )


def encode_tokens(
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    text_encoder1: CLIPTextModel | torch.nn.Module,
    text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
):
    # input_ids: b,77
    input_ids1 = input_ids1.to(text_encoder1.device)
    input_ids2 = input_ids2.to(text_encoder2.device)

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    last_hidden_state = enc_out["last_hidden_state"]
    pool2 = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids2.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),  # EOS token, assumed to be the max token id
    ]

    # apply projection
    pool2 = text_encoder2.text_projection(pool2.to(text_encoder2.text_projection.weight.dtype))
    pool2 = pool2.to(last_hidden_state.dtype)

    return hidden_states1, hidden_states2, pool2


if __name__ == "__main__":
    import sys
    import safetensors.torch
    from pixel_sdxl.model_utils import load_models_from_state_dict, serialize_models_to_state_dict
    import pixel_sdxl.train_utils as train_utils

    # test tokenizers and encoding
    ckpt = sys.argv[1]
    ckpt_to_save = sys.argv[2] if len(sys.argv) > 2 else None

    # Load the state dict
    state_dict = safetensors.torch.load(open(ckpt, "rb").read())

    text_encoder1, text_encoder2, unet, _ = load_models_from_state_dict(state_dict)
    text_encoder1.eval().to("cuda", dtype=torch.float16)
    text_encoder2.eval().to("cuda", dtype=torch.float16)

    tokenizer1, tokenizer2 = get_sdxl_tokenizers()
    text = "A beautiful painting of a sunset over the mountains."
    input_ids1, input_ids2 = tokenize(tokenizer1, tokenizer2, text)

    with torch.no_grad():
        hidden_states1, hidden_states2, pool2 = encode_tokens(input_ids1, input_ids2, text_encoder1, text_encoder2)
    print(f"hidden_states1: {hidden_states1.shape}, hidden_states2: {hidden_states2.shape}, pool2: {pool2.shape}")
    print(f"hidden_states1 dtype: {hidden_states1.dtype}, hidden_states2 dtype: {hidden_states2.dtype}, pool2 dtype: {pool2.dtype}")
    print(f"hidden_states1[0, :5, :5]: {hidden_states1[0, :5, :5]}")
    print(f"hidden_states2[0, :5, :5]: {hidden_states2[0, :5, :5]}")
    print(f"pool2[0, :5]: {pool2[0, :5]}")
    print(f"hidden_states1 mean: {hidden_states1.mean().item()}, std: {hidden_states1.std().item()}")
    print(f"hidden_states2 mean: {hidden_states2.mean().item()}, std: {hidden_states2.std().item()}")
    print(f"pool2 mean: {pool2.mean().item()}, std: {pool2.std().item()}")

    # test size embeddings
    vector = train_utils.get_size_embeddings(
        torch.tensor([[2560, 2048]]), torch.tensor([[0, 0]]), torch.tensor([[1280, 1024]]), device="cuda"
    )
    orig_emb = vector[:, :512]
    crop_emb = vector[:, 512:1024]
    target_emb = vector[:, 1024:]
    print(f"target_emb: {target_emb.shape}, orig_emb: {orig_emb.shape}, crop_emb: {crop_emb.shape}")
    print(f"mean/std orig_emb: {orig_emb.mean().item()}/{orig_emb.std().item()}")
    print(f"mean/std crop_emb: {crop_emb.mean().item()}/{crop_emb.std().item()}")
    print(f"mean/std target_emb: {target_emb.mean().item()}/{target_emb.std().item()}")

    if ckpt_to_save is not None:
        state_dict = serialize_models_to_state_dict(text_encoder1, text_encoder2, unet, None)
        safetensors.torch.save_file(state_dict, ckpt_to_save)
