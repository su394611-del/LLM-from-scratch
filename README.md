# Transformer Encoder from Scratch for Sentiment Classification

A practical PyTorch project that implements a **Transformer encoder from scratch** to classify customer reviews as **positive or negative**.

The project includes hand-written multi-head self-attention, sinusoidal positional encoding, pre-layer-normalization encoder blocks, feed-forward networks, residual connections, masked mean pooling, and a binary classification head. It also includes an optional optimized attention path using PyTorch's scaled dot-product attention (SDPA). **A new GPT-style decoder-only model will be built from scratch in the next iteration**.

> **Scope:** The Transformer encoder and classification head are trained from scratch. For real Yelp experiments, the input tokenizer and **frozen token-embedding matrix** are borrowed from [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). **A new small GPT-style decoder-only model will be built from scratch in the next iteration**.

## Features

- **Manual multi-head self-attention:** Separate query, key, and value projections, attention scaling, padding masks, and output projection.
- **Optional optimized attention:** Use `--sdpa` to enable `torch.nn.functional.scaled_dot_product_attention`.
- **Transformer encoder:** Configurable stack of pre-LayerNorm attention and GELU feed-forward blocks, with residual connections.
- **Position information:** Fixed sinusoidal positional encoding.
- **Sentiment classification:** Attention-mask-aware mean pooling and a two-class output layer.
- **Three execution modes:** Offline smoke test, small Yelp training run, and larger Yelp training run.
- **Training safeguards:** AdamW, gradient clipping, reproducible seed, validation-based checkpoint selection, and a held-out test split.
- **CPU or NVIDIA CUDA:** Automatically selects CUDA for real-data modes when the installed PyTorch build can access a GPU.
- **Optional diagram:** Export an architecture image with `torchview` and Graphviz.

## Project structure

```text
transformer-from-scratch/
├── README.md
├── transformer_from_scratch.py
├── requirements.txt
└── .gitignore
```

After training, a `transformer_checkpoints/` directory is created automatically. Checkpoint files are ignored by Git in the supplied `.gitignore`.

## Requirements

- Python **3.12 or newer** (Python 3.12 is a good starting choice on Windows).
- PyTorch `>=2.6` and tqdm `>=4.66` for the offline smoke test.
- Hugging Face Transformers `>=5.0,<6` and Datasets `>=4.0,<6` for real Yelp training.
- Internet access on the first real-data run to download the dataset and frozen MiniLM embedding weights, unless they are already cached.
- Optional: `torchview`, the Python `graphviz` package, and the **Graphviz system executable** to render a model diagram.

The included `requirements.txt` declares the base Python dependencies. Installing it may install a CPU-only PyTorch distribution; **GPU support depends on choosing a compatible CUDA-enabled PyTorch build** for your operating system, GPU, and driver.

## Installation (Windows PowerShell)

Clone the repository, or download and extract its files into a folder, then run:

```powershell
cd transformer-from-scratch

# Optional but recommended: isolated Python environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell prevents virtual-environment activation, you can invoke `.\.venv\Scripts\python.exe` directly instead.

If you are using the original two uploaded files rather than this packaged repository, your dependency filename may be `transformer_from_scratch_requirements(1).txt`. You can either install it directly:

```powershell
python -m pip install -r 'transformer_from_scratch_requirements(1).txt'
```

or rename it to `requirements.txt` before uploading to GitHub.

### Optional architecture visualization

```powershell
python -m pip install torchview graphviz
```

Graphviz may also need to be installed at the operating-system level for `.png` rendering.

## Quick start

### 1. Offline smoke test (no Hugging Face download)

```powershell
python transformer_from_scratch.py --smoke-test
```

Uses synthetic, balanced, Yelp-style text examples, a tiny in-memory tokenizer, and randomly initialized frozen embeddings. It automatically trains a small Transformer for **2 epochs** on **64 training**, **8 validation**, and **8 test** examples, using **CPU**. This is a code-correctness demonstration, **not** a meaningful sentiment benchmark.

### 2. Small Yelp Polarity experiment

```powershell
python transformer_from_scratch.py --train --sdpa
```

Downloads the [`fancyzhx/yelp_polarity`](https://huggingface.co/datasets/fancyzhx/yelp_polarity) dataset and MiniLM tokenizer/embedding weights if they are not already cached. Default requested split sizes are **1,000 training**, **200 validation**, and **200 test** examples, subject to the script's review-length filtering.

To customize the training configuration:

```powershell
python transformer_from_scratch.py --train --sdpa `
  --train-size 2000 --val-size 300 --test-size 300 `
  --batch-size 4 --max-len 128 --epochs 3 `
  --n-layers 3 --n-heads 6 --lr 0.0002
```

### 3. Larger Yelp experiment

```powershell
python transformer_from_scratch.py --full --sdpa
```

The `--full` preset requests **50,000 training**, **5,000 validation**, and **10,000 test** examples, with a **batch size of 8**, **256-token sequence length**, and **5 epochs**. These preset values override the corresponding CLI options. Running this on CPU may take substantial time; GPU memory requirements depend on the card and available VRAM.**Minimum VRAM: 8 GB; RAM on an Nvidia consumer-grade GPU is required**

### 4. Optional architecture diagram

```powershell
python transformer_from_scratch.py --smoke-test --visualize
```

If the visualization dependencies and Graphviz executable are available, the script writes `transformer_architecture.png` to the output directory. If not, it skips the optional diagram without failing training.

### 5. Custom checkpoint directory

```powershell
python transformer_from_scratch.py --train --sdpa --output .\runs\experiment_01
```

## Command-line options

| Option | Purpose | Default / notes |
| --- | --- | --- |
| `--smoke-test` | Small offline end-to-end test | Exactly one mode is required |
| `--train` | Smaller real Yelp training experiment | Exactly one mode is required |
| `--full` | Larger preset Yelp experiment | Exactly one mode is required |
| `--train-size` | Requested number of train examples | `1000` in `--train` |
| `--val-size` | Requested number of validation examples | `200` in `--train` |
| `--test-size` | Requested number of test examples | `200` in `--train` |
| `--batch-size` | Mini-batch size | `8` |
| `--max-len` | Maximum token sequence length | `128` |
| `--epochs` | Training epochs | `2` |
| `--n-layers` | Number of encoder blocks | `3` |
| `--n-heads` | Attention heads per encoder block | `6` |
| `--dropout` | Dropout probability | `0.05` |
| `--lr` | AdamW learning rate | `0.0002` |
| `--seed` | Random seed | `42` |
| `--sdpa` | Enable optimized scaled dot-product attention | Disabled unless specified |
| `--visualize` | Attempt to create a model diagram | Optional |
| `--output` | Output directory for checkpoint/diagram | `./transformer_checkpoints` beside the script |

See the full help page:

```powershell
python transformer_from_scratch.py --help
```

## Model architecture

```text
Review text
   │
   ▼
Tokenizer → token IDs + attention mask
   │
   ▼
Frozen token embedding lookup (MiniLM in real-data modes)
   │
   ▼
Embedding scale + sinusoidal positional encoding + dropout
   │
   ▼
┌──────────────── Transformer Encoder Block ────────────────┐
│ Pre-LayerNorm → Multi-head self-attention → residual add  │
│ Pre-LayerNorm → Feed-forward (GELU) → residual add        │
└──────────────────── repeat N times ───────────────────────┘
   │
   ▼
Final LayerNorm → masked mean pooling → Linear classifier
   │
   ▼
Logits for [negative, positive]
```

In the real-data path, MiniLM provides a **384-dimensional** frozen token embedding matrix. By default, the script builds **3 Transformer encoder blocks** with **6 attention heads** per block.

The encoder is implemented directly using PyTorch modules, rather than instantiating a pretrained Transformer encoder for the classifier.

## Training and evaluation

1. Load either synthetic smoke-test reviews or the Yelp Polarity dataset.
2. Tokenize and produce frozen token embeddings, plus padding/attention masks.
3. Train the custom Transformer with `AdamW` and cross-entropy loss.
4. Clip the gradient norm to `1.0` and report train/validation loss and accuracy.
5. Save the model checkpoint when validation accuracy improves.
6. Reload the best checkpoint and evaluate on the held-out test split **once**.

The checkpoint is saved by default to:

```text
transformer_checkpoints/best_transformer_static_emb.pt
```

The saved artifact contains the model state and training metadata, but **not the full frozen token-embedding matrix**. For real-model inference elsewhere, reuse the matching MiniLM tokenizer and embedding weights, and reconstruct the classifier configuration from the checkpoint metadata.

## Validation status

- Python source compilation: checked successfully.
- Offline smoke test: executed successfully on CPU/GPU with PyTorch `2.10.0+cpu`.
- The run completed both epochs, saved a checkpoint, reloaded it, and completed final test evaluation.
- The synthetic test's measured accuracy is **not a Yelp Polarity benchmark**.
- Large Yelp training and CUDA acceleration have been verified as part of this README preparation.

## Troubleshooting

**`ModuleNotFoundError`** — Confirm you're using the same Python interpreter in which dependencies were installed:

```powershell
python -c "import torch, tqdm, transformers, datasets; print('Imports OK')"
python -m pip check
```

**` torch.cuda.is_available()` is `False`** — The script will fall back to CPU/GPU for real-data training. Check the installed PyTorch CUDA build and the NVIDIA driver. A working GPU driver alone does not mean the currently installed PyTorch distribution supports CUDA.

**GPU out-of-memory** — Try `--train --batch-size 1 --max-len 64 --sdpa`, or run the smoke test on CPU—the `--full` preset overrides some batch and length parameters.

**Running from Jupyter** — The script requires one explicit mode argument and uses CLI parsing. Execute it through a notebook shell command, not by pasting its `argparse` main entry point into a cell:

```python
!python transformer_from_scratch.py --smoke-test
```

**First-run downloads fail** — The real-data modes use the Hugging Face Hub. Confirm internet connectivity or make sure the dataset and model are already cached.

## Dataset, model, and attribution

- Dataset: [Yelp Polarity on Hugging Face](https://huggingface.co/datasets/fancyzhx/yelp_polarity).
- Frozen tokenizer/embedding checkpoint: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Background: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [PyTorch scaled dot-product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

Consult the dataset and model pages for their respective licensing and usage conditions. This repository does not redistribute the dataset or pretrained model weights.

## License

This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0.

## Disclaimer

This project is intended for **Transformer internals and supervised text classification**.
