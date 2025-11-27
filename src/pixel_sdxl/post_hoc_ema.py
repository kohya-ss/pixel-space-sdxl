import torch
from safetensors.torch import load_file, save_file


def post_hoc_ema(ckpt_paths, ema_path):
    mu = 0.9  # 0.9〜0.99くらいで調整

    # 古い順に並んでいる前提で、後ろほど重みが大きいように
    weights = [mu ** (len(ckpt_paths) - 1 - i) for i in range(len(ckpt_paths))]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    avg_state = None

    for path, w in zip(ckpt_paths, weights):
        print(f"Loading checkpoint for EMA: {path} with weight {w:.6f}")
        if not path.endswith(".safetensors"):
            state = torch.load(path, map_location="cpu")
            if "model" in state:
                state = state["model"]
        else:
            state = load_file(path)

        if avg_state is None:
            avg_state = {k: v.float().clone() * w for k, v in state.items()}
        else:
            for k in avg_state.keys():
                avg_state[k] += state[k].float() * w

    print(f"Saving EMA checkpoint to: {ema_path}")
    save_file(avg_state, ema_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-hoc EMA for Pixel-SDXL checkpoints")
    parser.add_argument("--ckpt_paths", nargs="+", required=True, help="List of checkpoint paths to average (oldest to newest).")
    parser.add_argument("--ema_path", type=str, required=True, help="Output path for the EMA checkpoint.")
    args = parser.parse_args()

    if not args.ema_path.endswith(".safetensors"):
        raise ValueError("The output EMA path must end with .safetensors")

    post_hoc_ema(args.ckpt_paths, args.ema_path)
