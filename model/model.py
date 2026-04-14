import json
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------------------
# Tokenizer (tiny, from-scratch)
# ----------------------------
class SimpleTokenizer:
    """
    Whitespace tokenizer with a dataset-built vocab.
    """
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {i: w for w, i in vocab.items()}

    @staticmethod
    def build(texts: List[str], min_freq: int = 1) -> "SimpleTokenizer":
        from collections import Counter
        c = Counter()
        for t in texts:
            toks = t.lower().strip().split()
            c.update(toks)
        vocab = {SimpleTokenizer.PAD: 0, SimpleTokenizer.UNK: 1}
        for w, f in c.items():
            if f >= min_freq:
                vocab[w] = len(vocab)
        return SimpleTokenizer(vocab)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        toks = text.lower().strip().split()
        ids = [self.vocab.get(tok, self.vocab[self.UNK]) for tok in toks][:max_len]
        # pad
        if len(ids) < max_len:
            ids = ids + [self.vocab[self.PAD]] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ----------------------------
# Dataset
# ----------------------------
class DemoDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        text_json_path: str,
        tokenizer: SimpleTokenizer,
        image_size: int = 128,
        max_text_len: int = 16,
        joints_mean: Optional[np.ndarray] = None,
        joints_std: Optional[np.ndarray] = None,
    ):
        self.data = np.load(npz_path)
        self.images = self.data["images"]  # uint8 [N,H,W,3]
        self.joints = self.data["joints"].astype(np.float32)  # [N,J]

        with open(text_json_path, "r") as f:
            self.texts = json.load(f)["text"]
        assert len(self.texts) == len(self.images) == len(self.joints)

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_len = max_text_len

        # Normalize joints (helps a ton)
        if joints_mean is None or joints_std is None:
            joints_mean = self.joints.mean(axis=0)
            joints_std = self.joints.std(axis=0) + 1e-6
        self.joints_mean = joints_mean.astype(np.float32)
        self.joints_std = joints_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]  # uint8 HWC
        # Basic preprocessing: resize with simple nearest/bilinear via torch
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # CHW
        img_t = F.interpolate(img_t.unsqueeze(0), size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False).squeeze(0)

        text_ids = self.tokenizer.encode(self.texts[idx], self.max_text_len)

        joints = self.joints[idx]
        joints_norm = (joints - self.joints_mean) / self.joints_std
        joints_norm = torch.from_numpy(joints_norm)

        return img_t, text_ids, joints_norm


# ----------------------------
# Model
# ----------------------------
class TinyVisionEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        # Very small CNN
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        h = self.net(x).squeeze(-1).squeeze(-1)  # [B,128]
        return self.proj(h)  # [B,out_dim]


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B,T]
        e = self.emb(token_ids)  # [B,T,emb_dim]
        _, hT = self.gru(e)      # hT: [1,B,hidden_dim]
        h = hT.squeeze(0)        # [B,hidden_dim]
        return self.proj(h)


class TylerVLAPolicy(nn.Module):
    def __init__(self, vocab_size: int, num_joints: int, img_dim: int = 128, txt_dim: int = 128):
        super().__init__()
        self.vision = TinyVisionEncoder(out_dim=img_dim)
        self.text = TinyTextEncoder(vocab_size=vocab_size, emb_dim=64, hidden_dim=txt_dim)

        fusion_dim = img_dim + txt_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_joints),
        )

    def forward(self, img: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        v = self.vision(img)
        t = self.text(text_ids)
        z = torch.cat([v, t], dim=-1)
        return self.mlp(z)  # normalized joint targets
