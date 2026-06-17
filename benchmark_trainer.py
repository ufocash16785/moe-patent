"""
=============================================================================
  MoE Benchmark Trainer — TinyShakespeare + Math Operations
=============================================================================
  Purpose:  Train MoE transformer on two meaningful benchmarks to compare
            router variant effectiveness.

  Benchmark 1 — TinyShakespeare (character-level LM):
    - ~1MB Shakespeare corpus (karpathy/char-rnn)
    - Metric: perplexity per epoch
    - Patent angle: observe expert specialization by character type

  Benchmark 2 — Math Operations (synthetic):
    - "123+456=" → "579" (byte-level)
    - Difficulty: 2-digit, 4-digit, with carry
    - Metric: arithmetic accuracy per epoch
    - Patent angle: compare reasoning accuracy across router variants

  Usage:
    # Default: run both benchmarks with standard router
    python benchmark_trainer.py

    # Test a specific router variant
    python benchmark_trainer.py --router AttentionRouter

    # Run all router variants on both benchmarks
    python benchmark_trainer.py --all_variants

  Author:  Cash Chou / Scripta
  Date:    2026-06-17
=============================================================================
"""

import argparse
import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── Import from moe_model ────────────────────────────────────────────────────
from moe_model import (
    MoEConfig,
    MoERouter,
    SparseMoEBlock,
    MoETransformer,
)

# ── Import router variants ────────────────────────────────────────────────────
from router_variants import (
    AttentionRouter,
    EntropyAdaptiveRouter,
    GumbelRouter,
    SigmoidThresholdRouter,
    HierarchicalRouter,
    StatefulRouter,
    LoRAMoERouter,
    ContrastiveRouter,
)

ROUTER_VARIANTS = {
    "MoERouter": MoERouter,
    "AttentionRouter": AttentionRouter,
    "EntropyAdaptiveRouter": EntropyAdaptiveRouter,
    "GumbelRouter": GumbelRouter,
    "SigmoidThresholdRouter": SigmoidThresholdRouter,
    "HierarchicalRouter": HierarchicalRouter,
    "StatefulRouter": StatefulRouter,
    "ContrastiveRouter": ContrastiveRouter,
}

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  LARGER CONFIG DEFAULTS                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def make_large_config(vocab_size: int = 256, max_seq_len: int = 128, **kwargs) -> MoEConfig:
    """
    Scale-up config for meaningful MoE routing comparison.
    Old small config: d_model=64, num_layers=2, d_ff=128  → ~15K params
    New large config: d_model=256, num_layers=4, d_ff=512  → ~2-3M params
    """
    config = MoEConfig(
        # ── Vocabulary ──────────────────────────────────────────────
        vocab_size=vocab_size,      # byte-level: 256

        # ── Transformer — significantly larger ───────────────────────
        d_model=256,                # was 64
        num_heads=8,                # was 4
        d_ff=512,                   # was 128
        num_layers=4,               # was 2

        # ── MoE ─────────────────────────────────────────────────────
        num_experts=8,             # total experts
        top_k=2,                   # selected per token

        # ── Training ─────────────────────────────────────────────────
        max_seq_len=max_seq_len,    # was 32
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=32,             # was 8
        num_steps=2000,            # was 200

        # ── Device ──────────────────────────────────────────────────
        device="cpu",
    )

    for k, v in kwargs.items():
        setattr(config, k, v)

    return config


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DATASET 1 — TinyShakespeare (Character-Level LM)                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class TinyShakespeareDataset(Dataset):
    """
    Character-level Shakespeare dataset.
    Each sample is a block of text of length max_seq_len.
    Target = same text shifted by 1 (next-character prediction).
    """
    def __init__(self, data: bytes, max_seq_len: int, stride: int = None):
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        self.data = data
        self.total_samples = max(1, (len(data) - max_seq_len) // self.stride)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start: start + self.max_seq_len]
        # Pad if needed
        if len(chunk) < self.max_seq_len:
            chunk = chunk + b'\x00' * (self.max_seq_len - len(chunk))

        x = torch.tensor(list(chunk), dtype=torch.long)
        y = x.clone()
        return x, y


def download_shakespeare() -> bytes:
    cache = Path.home() / ".cache" / "char-rnn"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "tinyshakespeare.txt"

    if path.exists():
        print(f"  [cache hit] {path}")
        return path.read_bytes()

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"  Downloading TinyShakespeare from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")
    return path.read_bytes()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DATASET 2 — Math Operations (Synthetic)                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class MathDataset(Dataset):
    """
    Synthetic arithmetic dataset.
    Format: "123+456=" → "579"
    Each character is one byte (ASCII digit, '+', '-', '*', '=').

    Difficulty levels:
      easy:   2-digit + 2-digit (e.g., 12+34=)
      medium: 4-digit + 4-digit (e.g., 1234+5678=)
      hard:   4-digit with carry (e.g., 9999+0001=)
    """
    def __init__(self, size: int, max_seq_len: int, difficulty: str = "medium"):
        self.size = size
        self.max_seq_len = max_seq_len
        self.difficulty = difficulty

    def __len__(self) -> int:
        return self.size

    def _generate_sample(self):
        if self.difficulty == "easy":
            a = torch.randint(10, 100, (1,)).item()
            b = torch.randint(10, 100, (1,)).item()
        elif self.difficulty == "hard":
            a = torch.randint(1, 10000, (1,)).item()
            b = torch.randint(1, 10000, (1,)).item()
        else:  # medium (default)
            a = torch.randint(100, 10000, (1,)).item()
            b = torch.randint(100, 10000, (1,)).item()

        result = a + b
        s = f"{a}+{b}="
        r = str(result)

        # Encode as bytes
        inp = s.encode('ascii')
        out = r.encode('ascii')

        # Truncate to max_seq_len
        if len(inp) > self.max_seq_len:
            inp = inp[:self.max_seq_len]
        if len(out) > self.max_seq_len:
            out = out[:self.max_seq_len]

        return inp, out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(idx * 31337)
        inp, out = self._generate_sample()

        # Pad to max_seq_len
        x = list(inp) + [0] * (self.max_seq_len - len(inp))
        y = list(out) + [0] * (self.max_seq_len - len(out))

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING LOOP                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def compute_perplexity(loss: float) -> float:
    return math.exp(loss)


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Character-level accuracy: fraction of positions where predicted char == target."""
    preds = logits.argmax(dim=-1)           # [batch, seq]
    mask = targets != 0                      # ignore pad token
    correct = (preds == targets) & mask
    return correct.float().sum().item() / mask.float().sum().item()


@torch.no_grad()
def compute_math_accuracy(model: MoETransformer, dataset: MathDataset,
                           config: MoEConfig, num_samples: int = 200) -> float:
    """
    Compute exact-string accuracy on math problems.
    A prediction is correct only if the entire output string matches.
    """
    model.eval()
    correct = 0
    total = 0

    for i in range(min(num_samples, len(dataset))):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(config.device)
        y = dataset[i][1]  # target bytes

        # Generate
        output = model.generate(x, max_new=dataset.max_seq_len)
        pred = output[0, x.size(1):].tolist()

        # Strip padding (0) and compare
        pred_str = bytes(pred).rstrip(b'\x00').decode('ascii', errors='ignore')
        target_str = bytes(y.tolist()).rstrip(b'\x00').decode('ascii', errors='ignore')

        if pred_str == target_str:
            correct += 1
        total += 1

    model.train()
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def analyse_expert_specialization(model: MoETransformer, config: MoEConfig,
                                   dataset, num_batches: int = 10) -> Dict:
    """
    For TinyShakespeare: analyse which experts specialize on which character types.
    For Math: analyse which experts handle digits vs operators.
    Returns per-expert byte distribution.
    """
    model.eval()
    expert_char_counts = {i: {} for i in range(config.num_experts)}

    for batch_idx in range(num_batches):
        if hasattr(dataset, '__getitem__'):
            x, _ = dataset[batch_idx % len(dataset)]
            x = x.unsqueeze(0).to(config.device)
        else:
            x = dataset()
            x = x.to(config.device)

        # Get hidden states at MoE layer
        hidden = model.token_embedding(x)
        if hidden.size(1) > config.max_seq_len:
            hidden = hidden[:, :config.max_seq_len, :]
        hidden = model.layers[0].ln2(hidden)

        with torch.no_grad():
            _, expert_indices, _ = model.layers[0].moe.router(hidden)
            # expert_indices: [1*seq, top_k]

        expert_indices = expert_indices.view(-1, config.top_k)[:, 0]  # top-1

        # Get input bytes
        chars = x.view(-1).cpu().tolist()

        for char, expert in zip(chars, expert_indices.view(-1).cpu().tolist()):
            expert = int(expert)
            char = chr(char) if char < 128 else '.'
            expert_char_counts[expert][char] = expert_char_counts[expert].get(char, 0) + 1

    model.train()

    # Find top chars per expert
    summary = {}
    for expert, counts in expert_char_counts.items():
        if counts:
            sorted_chars = sorted(counts.items(), key=lambda x: -x[1])[:5]
            summary[expert] = {c: n for c, n in sorted_chars}
    return summary


def train_benchmark(
    model: MoETransformer,
    config: MoEConfig,
    dataset,
    dataset_name: str,
    is_math: bool = False,
    math_dataset: Optional[MathDataset] = None,
    eval_every: int = 100,
    num_eval_samples: int = 200,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Train model on a dataset and track metrics over time.
    Returns: {"steps": [], "loss": [], "perplexity": [], "accuracy": [], "aux_loss": []}
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_steps, eta_min=1e-5
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    metrics = {"step": [], "loss": [], "perplexity": [], "accuracy": [], "aux_loss": []}

    step = 0
    data_iter = iter(dataloader)

    while step < config.num_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(config.device), y.to(config.device)

        logits, aux_loss = model(x)
        ce_loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1),
            ignore_index=0,
        )
        total_loss = ce_loss + config.router_aux_loss_coef * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_logits, val_aux = model(x)
                val_ce = F.cross_entropy(
                    val_logits.view(-1, config.vocab_size),
                    y.view(-1),
                    ignore_index=0,
                )
                perplexity = compute_perplexity(val_ce.item())
                accuracy = compute_accuracy(val_logits, y)

            model.train()

            metrics["step"].append(step)
            metrics["loss"].append(val_ce.item())
            metrics["perplexity"].append(perplexity)
            metrics["accuracy"].append(accuracy)
            metrics["aux_loss"].append(val_aux.item())

            if verbose:
                if is_math:
                    math_acc = compute_math_accuracy(model, math_dataset, config, num_eval_samples)
                    print(f"  Step {step:5d} | Loss: {val_ce.item():.4f} | "
                          f"CharAcc: {accuracy:.4f} | MathAcc: {math_acc:.4f} | "
                          f"Aux: {val_aux.item():.4f} | PPL: {perplexity:.2f}")
                else:
                    print(f"  Step {step:5d} | Loss: {val_ce.item():.4f} | "
                          f"Acc: {accuracy:.4f} | Aux: {val_aux.item():.4f} | PPL: {perplexity:.2f}")

    return metrics


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER ANALYSIS PRINTER                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def print_router_analysis(model: MoETransformer, config: MoEConfig,
                           dataset, num_batches: int = 10):
    """Print expert load balance and specialization."""
    model.eval()

    expert_counts = torch.zeros(config.num_experts)
    total_tokens = 0

    for batch_idx in range(num_batches):
        x, _ = dataset[batch_idx % len(dataset)]
        x = x.unsqueeze(0).to(config.device)

        hidden = model.token_embedding(x)
        hidden = model.layers[0].ln2(hidden)
        _, expert_indices, _ = model.layers[0].moe.router(hidden)
        expert_indices = expert_indices.view(-1)

        for i in range(config.num_experts):
            expert_counts[i] += (expert_indices == i).sum().item()
        total_tokens += x.numel() * config.top_k

    model.train()

    print(f"\n  Expert token counts (top-1): {expert_counts.tolist()}")
    ideal = total_tokens / config.num_experts
    imbalance = (expert_counts - ideal).abs().sum().item() / ideal
    print(f"  Ideal per expert: {ideal:.0f}  |  Imbalance ratio: {imbalance:.4f}")

    # Show specialization
    spec = analyse_expert_specialization(model, config, dataset, num_batches=5)
    print(f"\n  Top chars per expert (layer 1):")
    for expert, chars in spec.items():
        top = ', '.join(f"'{c}'({n})" for c, n in chars.items())
        print(f"    Expert {expert}: {top}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def run_shakespeare(router_cls=None, config=None, verbose=True) -> Tuple[MoETransformer, Dict]:
    """Run TinyShakespeare benchmark."""
    print("\n" + "="*60)
    print("  Benchmark 1 — TinyShakespeare (Character-Level LM)")
    print("="*60)

    if config is None:
        config = make_large_config(max_seq_len=128)

    if router_cls is not None:
        SparseMoEBlock.router_class = router_cls

    # Download data
    print("\n[Shakespeare Data]")
    data = download_shakespeare()
    print(f"  Dataset size: {len(data):,} bytes  |  Vocab: byte-level (256)")

    # Dataset — use stride for more training samples
    dataset = TinyShakespeareDataset(data, config.max_seq_len, stride=64)
    print(f"  Samples: {len(dataset):,}  (seq_len={config.max_seq_len}, stride=64)")

    model = MoETransformer(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}  |  Router: {SparseMoEBlock.router_class.__name__}")

    print("\n[Training]")
    metrics = train_benchmark(
        model, config, dataset, dataset_name="Shakespeare",
        eval_every=200, verbose=verbose
    )

    print("\n[Router Analysis]")
    print_router_analysis(model, config, dataset, num_batches=10)

    return model, metrics


def run_math(router_cls=None, config=None, verbose=True) -> Tuple[MoETransformer, Dict]:
    """Run Math Operations benchmark."""
    print("\n" + "="*60)
    print("  Benchmark 2 — Math Operations (Synthetic)")
    print("="*60)

    if config is None:
        config = make_large_config(max_seq_len=32)  # math seqs are short

    if router_cls is not None:
        SparseMoEBlock.router_class = router_cls

    # Create datasets for each difficulty
    train_dataset = MathDataset(size=10000, max_seq_len=config.max_seq_len, difficulty="medium")
    eval_dataset  = MathDataset(size=500,   max_seq_len=config.max_seq_len, difficulty="medium")

    print(f"\n  Difficulty: medium (3-4 digit addition)")
    print(f"  Train samples: {len(train_dataset):,}  |  Eval samples: {len(eval_dataset):,}")
    print(f"  Seq len: {config.max_seq_len}")

    model = MoETransformer(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}  |  Router: {SparseMoEBlock.router_class.__name__}")

    print("\n[Training]")
    metrics = train_benchmark(
        model, config, train_dataset,
        dataset_name="Math",
        is_math=True,
        math_dataset=eval_dataset,
        eval_every=200,
        verbose=verbose,
    )

    # Final math accuracy
    print("\n[Final Math Accuracy]")
    final_acc = compute_math_accuracy(model, eval_dataset, config, num_samples=500)
    print(f"  Exact string accuracy: {final_acc:.4f} ({int(final_acc*500)}/500)")

    return model, metrics


def run_all_variants(shakespeare_only=False, math_only=False):
    """
    Run both benchmarks with ALL router variants.
    Prints a comparison table at the end.
    """
    results = {}

    for name, RouterCls in ROUTER_VARIANTS.items():
        print(f"\n{'#'*70}")
        print(f"#  ROUTER: {name}")
        print(f"{'#'*70}")

        # Reset to standard router class
        SparseMoEBlock.router_class = RouterCls

        try:
            if not math_only:
                _, sh_metrics = run_shakespeare(router_cls=RouterCls, verbose=False)
                final_ppl = sh_metrics["perplexity"][-1]
                final_acc = sh_metrics["accuracy"][-1]
                print(f"  Shakespeare Final → PPL: {final_ppl:.2f}  |  CharAcc: {final_acc:.4f}")
            else:
                final_ppl, final_acc = None, None

            if not shakespeare_only:
                _, math_metrics = run_math(router_cls=RouterCls, verbose=False)
                final_math_acc = math_metrics["accuracy"][-1]
                print(f"  Math Final        → CharAcc: {final_math_acc:.4f}")
            else:
                final_math_acc = None

            results[name] = {
                "shakespeare_ppl": final_ppl,
                "shakespeare_acc": final_acc,
                "math_acc": final_math_acc,
            }
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results[name] = {"error": str(e)}

    # ── Comparison Table ──────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  ROUTER COMPARISON SUMMARY")
    print("="*70)
    print(f"  {'Router':<30} {'Shakespeare PPL':>16} {'Shakespeare Acc':>16} {'Math Acc':>10}")
    print("-" * 70)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<30} {'FAIL':>16}  {'—':>16}  {'—':>10}")
        else:
            ppl = f"{r['shakespeare_ppl']:.2f}" if r['shakespeare_ppl'] else "—"
            s_acc = f"{r['shakespeare_acc']:.4f}" if r['shakespeare_acc'] else "—"
            m_acc = f"{r['math_acc']:.4f}" if r['math_acc'] else "—"
            print(f"  {name:<30} {ppl:>16}  {s_acc:>16}  {m_acc:>10}")

    return results


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Benchmark Trainer")
    parser.add_argument("--router", type=str, default=None,
                        help="Router class name to use (default: standard MoERouter)")
    parser.add_argument("--all_variants", action="store_true",
                        help="Run all router variants on both benchmarks")
    parser.add_argument("--shakespeare_only", action="store_true")
    parser.add_argument("--math_only", action="store_true")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all_variants:
        results = run_all_variants(
            shakespeare_only=args.shakespeare_only,
            math_only=args.math_only,
        )
    else:
        # Determine router
        router_cls = None
        if args.router:
            if args.router not in ROUTER_VARIANTS:
                print(f"Unknown router: {args.router}")
                print(f"Available: {list(ROUTER_VARIANTS.keys())}")
                exit(1)
            router_cls = ROUTER_VARIANTS[args.router]
            SparseMoEBlock.router_class = router_cls
            print(f"Using router: {args.router}")
        else:
            print("Using standard MoERouter")

        # Config with overrides
        config = make_large_config(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
        )
        config.device = device

        if not args.math_only:
            run_shakespeare(router_cls=router_cls, config=config)

        if not args.shakespeare_only:
            run_math(router_cls=router_cls, config=config)

    print("\n✅ Done")
