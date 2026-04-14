import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import SimpleTokenizer, TylerVLAPolicy


def load_policy(run_dir: str, device: str = None):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    with open(run_dir / "tokenizer.json", "r") as f:
        vocab = json.load(f)["vocab"]
    tokenizer = SimpleTokenizer(vocab)
    norm = np.load(run_dir / "joint_norm.npz")
    j_mean = norm["mean"].astype(np.float32)
    j_std = norm["std"].astype(np.float32)

    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = TylerVLAPolicy(vocab_size=ckpt["vocab_size"], num_joints=ckpt["num_joints"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, j_mean, j_std, device


def preprocess_image(img_hwc_uint8: np.ndarray, image_size: int = 128) -> torch.Tensor:
    # img: HWC uint8 -> torch CHW float [0,1]
    x = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0
    x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size),
                      mode="bilinear", align_corners=False).squeeze(0)
    return x
