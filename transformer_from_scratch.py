"""Transformer text classifier from scratch: corrected for Windows / Python 3.12+.

Based on the user's Yelp Polarity example. The Transformer encoder and attention
layers are implemented directly in PyTorch; only tokenization and *frozen token
embeddings* are borrowed from the MiniLM checkpoint.

Modes:
    python transformer_from_scratch_corrected.py --smoke-test
    python transformer_from_scratch_corrected.py --train
    python transformer_from_scratch_corrected.py --full

--smoke-test uses synthetic text, an in-memory tokenizer, and random frozen
embeddings, so it does NOT need datasets, transformers, or internet access.
--train and --full use the actual Hugging Face Yelp Polarity dataset and the
MiniLM token-embedding matrix. They require a Hugging Face download if not cached.

The output is a *binary sentiment classifier*, not a generative LLM.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_ID = "fancyzhx/yelp_polarity"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "transformer_checkpoints"


class YelpTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts, self.labels = texts, labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def make_collate_with_static_embeddings(tokenizer, emb_weight: torch.Tensor, max_len: int):
    """Tokenize on CPU and look up frozen, non-contextualized wordpiece embeddings."""
    if emb_weight.device.type != "cpu":
        raise ValueError("Frozen embeddings must be on the CPU for DataLoader collating")

    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()
        if input_ids.min().item() < 0 or input_ids.max().item() >= emb_weight.shape[0]:
            raise ValueError("Tokenizer vocabulary doesn't match the embedding matrix")
        return {
            "embeddings": F.embedding(input_ids, emb_weight),
            "attention_mask": attention_mask,
            "pad_mask": attention_mask.eq(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2).float()
                          * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * scale)
        # Support odd d_model as well as even d_model.
        pe[:, 1::2] = torch.cos(positions * scale[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds positional-encoding max_len")
        return x + self.pe[:, :x.size(1)].to(dtype=x.dtype)


class MultiHeadSelfAttention(nn.Module):
    """Explicit Q/K/V attention for learning; optional PyTorch SDPA for lower memory."""
    def __init__(self, d_model: int, n_heads: int = 6,
                 dropout: float = 0.1, use_sdpa: bool = False):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_sdpa = use_sdpa

    def forward(self, x: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        # Boolean 'True' in the SDPA mask means the key is allowed to attend.
        keep_keys = ~pad_mask[:, None, None, :] if pad_mask is not None else None
        if self.use_sdpa:
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep_keys,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if pad_mask is not None:
                scores = scores.masked_fill(pad_mask[:, None, None, :],
                                            torch.finfo(scores.dtype).min)
            attention = self.dropout(torch.softmax(scores, dim=-1))
            context = attention @ v
        context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 6,
                 d_ff: Optional[int] = None, dropout: float = 0.1,
                 use_sdpa: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_sdpa)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff or 4 * d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff or 4 * d_model, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.ln1(x), pad_mask))
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x


class TransformerClassifier(nn.Module):
    """Three (configurable) encoder blocks, masked mean pooling, binary classification."""
    def __init__(self, d_model: int, n_heads: int = 6, n_layers: int = 3,
                 max_len: int = 256, num_classes: int = 2,
                 dropout: float = 0.05, use_sdpa: bool = False):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout=dropout, use_sdpa=use_sdpa)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x_emb: torch.Tensor, attention_mask: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        # Keep the book's sqrt(D) embedding scale: learnable blocks remain from scratch.
        x = self.dropout(self.pos_enc(x_emb * math.sqrt(x_emb.shape[-1])))
        for layer in self.layers:
            x = layer(x, pad_mask)
        x = self.ln_f(x)
        weights = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.head(pooled)


def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["embeddings"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["pad_mask"].to(device, non_blocking=True),
            )
            labels = batch["labels"].to(device, non_blocking=True)
            count = labels.numel()
            loss_sum += F.cross_entropy(logits, labels).item() * count
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += count
    return loss_sum / max(total, 1), correct / max(total, 1)


class SmokeTokenizer:
    """Tiny offline tokenizer solely for end-to-end code verification."""
    def __init__(self):
        words = ["[PAD]", "[UNK]", "great", "excellent", "delicious", "good",
                 "amazing", "perfect", "terrible", "awful", "bad", "horrible",
                 "poor", "disappointing", "food", "service", "restaurant", "meal",
                 "the", "was", "really", "quite", "and", "experience"]
        self.vocab = {word: i for i, word in enumerate(words)}

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert return_tensors == "pt"
        ids = []
        for text in texts:
            tokens = [self.vocab.get(word.strip(".,!?"), 1) for word in text.lower().split()]
            ids.append(tokens[:max_length])
        masks = [[1] * len(seq) + [0] * (max_length - len(seq)) for seq in ids]
        padded = [seq + [0] * (max_length - len(seq)) for seq in ids]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}


def make_smoke_data():
    # Balanced, synthetic Yelp-style examples, not real Yelp data.
    rng = random.Random(42)
    positive = ["great", "excellent", "delicious", "good", "amazing", "perfect"]
    negative = ["terrible", "awful", "bad", "horrible", "poor", "disappointing"]
    contexts = ["the food was", "the service was", "the meal was", "the restaurant was"]
    samples = []
    for i in range(80):
        label = i % 2
        words = positive if label else negative
        samples.append((f"{rng.choice(contexts)} really {rng.choice(words)} and {rng.choice(words)}", label))
    rng.shuffle(samples)
    return samples[:64], samples[64:72], samples[72:]


def select_reviews(hf_split, count: int, char_limit: int, seed: int):
    """Shuffle then filter a bounded candidate pool to avoid costly full-dataset filtering."""
    if count < 1:
        raise ValueError("Requested sample count must be positive")
    shuffled = hf_split.shuffle(seed=seed)
    # Short reviews are common; sample 3x requested to allow length filtering.
    candidate_count = min(len(shuffled), max(count * 3, count + 100))
    candidates = shuffled.select(range(candidate_count))
    candidates = candidates.filter(lambda row: bool(row["text"].strip())
                                   and len(row["text"]) <= char_limit)
    if len(candidates) < count:
        print(f"NOTE: found {len(candidates)} eligible examples out of {candidate_count} candidates")
    final = candidates.select(range(min(count, len(candidates))))
    return final["text"], [int(label) for label in final["label"]]


def load_real_data(train_count: int, val_count: int,
                   test_count: int, char_limit: int):
    from datasets import load_dataset  # Import only in real-data mode.
    print("Loading Yelp Polarity dataset (downloads may be required)...", flush=True)
    all_data = load_dataset(DATASET_ID)
    # Validation is drawn from the original training split; test split is held out.
    train_texts, train_labels = select_reviews(
        all_data["train"], train_count + val_count, char_limit, seed=42
    )
    if len(train_texts) <= val_count:
        raise ValueError("Not enough filtered training examples for the validation split")
    split = len(train_texts) - val_count
    train_data = list(zip(train_texts[:split], train_labels[:split]))
    val_data = list(zip(train_texts[split:], train_labels[split:]))
    test_texts, test_labels = select_reviews(all_data["test"], test_count, char_limit, seed=43)
    test_data = list(zip(test_texts, test_labels))
    return train_data, val_data, test_data


def try_visualize(model: nn.Module, output_dir: Path,
                  max_len: int, d_model: int) -> None:
    """Optional rendering; absent torchview/Graphviz must not prevent training."""
    try:
        from torchview import draw_graph
        # A separate CPU example avoids temporarily moving the actual training model.
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            mask = torch.ones(1, max_len, dtype=torch.long, device=device)
            mask[:, -min(4, max_len):] = 0
            graph = draw_graph(
                model, input_data=(
                    torch.randn(1, max_len, d_model, device=device),
                    mask, mask.eq(0)),
                depth=2, expand_nested=False, graph_name="TransformerClassifier",
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            graph.visual_graph.render(
                filename="transformer_architecture", directory=str(output_dir),
                format="png", view=False, cleanup=True,
            )
        print("Architecture image saved to", output_dir / "transformer_architecture.png")
    except Exception as exc:
        print(f"Optional architecture diagram skipped: {type(exc).__name__}: {exc}")


def run(args: argparse.Namespace):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke_test:
        # Make synthetic test execution predictable and fast on CPU-only machines.
        torch.set_num_threads(min(2, torch.get_num_threads()))
        args.max_len, args.batch_size, args.epochs = 24, 8, 2
        args.n_layers, args.n_heads = 2, 4
        train_data, val_data, test_data = make_smoke_data()
        tokenizer = SmokeTokenizer()
        emb_weight = torch.randn(len(tokenizer.vocab), 48) * 0.1
        device = torch.device("cpu")
        print("Mode: OFFLINE SMOKE TEST (synthetic data, no Hugging Face downloads)")
    else:
        if args.full:
            args.train_size, args.val_size, args.test_size = 50_000, 5_000, 10_000
            args.batch_size, args.max_len, args.epochs = 8, 256, 5
        train_data, val_data, test_data = load_real_data(
            args.train_size, args.val_size, args.test_size, args.max_len * 6
        )
        from transformers import AutoModel, AutoTokenizer  # Lazy import.
        print("Loading pretrained tokenizer and FROZEN input embeddings:", MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        backbone = AutoModel.from_pretrained(MODEL_ID)
        input_layer = backbone.get_input_embeddings()
        if input_layer is None:
            raise RuntimeError("The pretrained model has no token embeddings")
        emb_weight = input_layer.weight.detach().cpu().clone()
        del backbone
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("WARNING: CUDA is not available in THIS Python environment; CPU training may be slow.")
        else:
            print("GPU:", torch.cuda.get_device_name(0))

    if not train_data or not val_data or not test_data:
        raise ValueError("One or more dataset splits are empty")
    print(f"Dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"Device: {device} | torch={torch.__version__}")
    collate = make_collate_with_static_embeddings(tokenizer, emb_weight, args.max_len)
    def loader(samples, shuffle):
        return DataLoader(
            YelpTextDataset([x[0] for x in samples], [x[1] for x in samples]),
            batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=collate, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    train_loader, val_loader, test_loader = (
        loader(train_data, True), loader(val_data, False), loader(test_data, False)
    )
    d_model = emb_weight.shape[1]
    model = TransformerClassifier(
        d_model=d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        max_len=args.max_len, num_classes=2, dropout=args.dropout,
        use_sdpa=args.sdpa,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Transformer parameters: {total_params:,}")

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        try_visualize(model, out_dir, args.max_len, d_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_acc, best_path = -1.0, out_dir / "best_transformer_static_emb.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = batch["embeddings"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, attention_mask, pad_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            seen += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={loss_sum/seen:.4f} "
              f"train_accuracy={correct/seen:.4f} | "
              f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}", flush=True)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save({
                "model_state": model.state_dict(), "best_validation_accuracy": best_acc,
                "epoch": epoch, "d_model": d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "max_len": args.max_len,
                "num_classes": 2, "dropout": args.dropout, "use_sdpa": args.sdpa,
                "tokenizer_id": "synthetic" if args.smoke_test else MODEL_ID,
            }, best_path)
            print("Saved checkpoint:", best_path)

    # Evaluate official test split only ONCE, after best validation checkpoint is selected.
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"FINAL: best_val_accuracy={best_acc:.4f} "
          f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")
    print("NOTE: SMOKE TEST figures are on synthetic data." if args.smoke_test
          else "NOTE: Results depend on downloaded Yelp data and chosen subset sizes.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="Quick OFFLINE end-to-end training")
    mode.add_argument("--train", action="store_true", help="Small real Yelp training run")
    mode.add_argument("--full", action="store_true", help="Large original-style Yelp training run")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sdpa", action="store_true", help="Use PyTorch efficient scaled dot-product attention")
    parser.add_argument("--visualize", action="store_true", help="Optionally render torchview diagram")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
