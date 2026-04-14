import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.model import DemoDataset, SimpleTokenizer, TylerVLAPolicy


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    image_size: int = 128
    max_text_len: int = 16
    val_split: float = 0.1
    seed: int = 0


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(
    npz_path: str,
    text_json_path: str,
    out_dir: str | Path = "runs/tyler_vla",
    cfg: TrainConfig = TrainConfig(),
):
    set_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load texts to build vocab
    with open(text_json_path, "r") as f:
        texts = json.load(f)["text"]
    tokenizer = SimpleTokenizer.build(texts, min_freq=1)

    # Load npz once to get joint dims + compute normalization
    arr = np.load(npz_path)
    joints = arr["joints"].astype(np.float32)
    num_joints = joints.shape[1]
    joints_mean = joints.mean(axis=0)
    joints_std = joints.std(axis=0) + 1e-6

    # Save tokenizer + joint stats
    with open(out_dir / "tokenizer.json", "w") as f:
        json.dump({"vocab": tokenizer.vocab}, f)
    np.savez(out_dir / "joint_norm.npz", mean=joints_mean, std=joints_std)

    ds = DemoDataset(
        npz_path=npz_path,
        text_json_path=text_json_path,
        tokenizer=tokenizer,
        image_size=cfg.image_size,
        max_text_len=cfg.max_text_len,
        joints_mean=joints_mean,
        joints_std=joints_std,
    )

    # Split
    N = len(ds)
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = int(N * cfg.val_split)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_ds = torch.utils.data.Subset(ds, tr_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = TylerVLAPolicy(vocab_size=len(tokenizer.vocab), num_joints=num_joints).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    for ep in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for img, text_ids, joints_norm in tr_loader:
            img = img.to(cfg.device)
            text_ids = text_ids.to(cfg.device)
            joints_norm = joints_norm.to(cfg.device)

            pred = model(img, text_ids)
            loss = F.mse_loss(pred, joints_norm)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()

        tr_loss /= max(1, len(tr_loader))

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, text_ids, joints_norm in val_loader:
                img = img.to(cfg.device)
                text_ids = text_ids.to(cfg.device)
                joints_norm = joints_norm.to(cfg.device)
                pred = model(img, text_ids)
                val_loss += F.mse_loss(pred, joints_norm).item()
        val_loss /= max(1, len(val_loader))

        print(f"epoch {ep+1:03d} | train {tr_loss:.6f} | val {val_loss:.6f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_joints": num_joints,
                    "vocab_size": len(tokenizer.vocab),
                    "cfg": cfg.__dict__,
                },
                out_dir / "model.pt",
            )

    print(f"done. best val: {best_val:.6f}. saved to: {out_dir/'model.pt'}")


if __name__ == "__main__":
    # Example:
    # python -c "from model import train; train('demos/merged.npz', 'demos/merged.json', 'runs/pick_place_v1')"
    pass
