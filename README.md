# MoE Router Patent Strategy Guide

**Location**: `~/docs/moe-patent/`
**Date**: 2026-06-17

## Files

| File | Description |
|------|-------------|
| `moe_model.py` | Full MoE transformer with standard top-2 router (~2-3M params with large config) |
| `router_variants.py` | 8 alternative router designs you can swap in |
| `benchmark_trainer.py` | **NEW** — TinyShakespeare + Math benchmarks for router comparison |

## Quick Start

```bash
cd ~/docs/moe-patent

# Run both benchmarks with standard router
python benchmark_trainer.py

# Run all 8 router variants on both benchmarks (takes ~30 min)
python benchmark_trainer.py --all_variants

# Test a specific router variant
python benchmark_trainer.py --router AttentionRouter

# Customize model size
python benchmark_trainer.py --d_model 512 --num_layers 6 --num_steps 5000
```

## Benchmark 1 — TinyShakespeare (Character-Level LM)

- ~1MB Shakespeare corpus (karpathy/char-rnn)
- Auto-downloads on first run (~1MB)
- **Metric**: Perplexity per step (lower = better)
- **Patent angle**: observe expert specialization by character type (vowels, consonants, punctuation, spaces)
- **Byte vocab**: 256 (each ASCII char = 1 byte)
- **Default config**: d_model=256, num_layers=4, d_ff=512, seq_len=128

## Benchmark 2 — Math Operations (Synthetic)

- Format: `123+456=` → `579` (byte-level, ASCII digits)
- **Metric**: Exact string accuracy (full output must match)
- **Patent angle**: direct reasoning accuracy comparison across router variants
- **Difficulty**: medium (3-4 digit addition), no division to avoid float issues
- **Default config**: d_model=256, num_layers=4, seq_len=32

## How to Swap a Router Variant

**Method 1 — Class attribute (recommended):**
```python
from router_variants import AttentionRouter
from moe_model import SparseMoEBlock

SparseMoEBlock.router_class = AttentionRouter
model = MoETransformer(config)
```

**Method 2 — Via CLI:**
```bash
python benchmark_trainer.py --router HierarchicalRouter
```

## Architecture Overview

```
 hidden_states [batch, seq, d_model]
       │
   LayerNorm
       │
 CausalSelfAttention (num_heads=8, d_model=256)
       │ residual (+)
   LayerNorm
       │
   ROUTER (the key part)
 Linear(d_model, num_experts) → softmax → top-k
       │
 ┌─────┴─────┐
 │ Expert 0  │ Expert 1 ... Expert 7
 │  (MLP)    │  each: Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)
 └─────┬─────┘
       │ weighted sum
  final_output
```

## Router Variants & Patent Angles

| # | Variant | Key Idea | Patent Angle |
|---|---------|----------|--------------|
| A | **Sigmoid + Threshold** | Dynamic k per token via sigmoid threshold | Adaptive compute budget |
| B | **Hierarchical Router** | Coarse cluster → fine expert selection | Reduced routing cost |
| C | **Entropy-Adaptive Top-k** | k depends on routing confidence | Per-token expert count |
| D | **Attention-Based** | Expert embeddings + dot-product routing | Content-expert matching |
| E | **Gumbel-Softmax** | Stochastic routing during training | Exploration in routing |
| F | **Stateful/Recurrent** | Previous routing feeds current decision | Temporal consistency |
| G | **LoRA-Style Experts** | Low-rank adapters as experts | Parameter-efficient MoE |
| H | **Contrastive Router** | Diversity-preserving routing loss | Expert specialization |

## Comparison Table (run `--all_variants` to populate)

```
Router                Shakespeare PPL  Shakespeare Acc  Math Acc
─────────────────────────────────────────────────────────────────
MoERouter (baseline)
AttentionRouter
EntropyAdaptiveRouter
GumbelRouter
SigmoidThresholdRouter
HierarchicalRouter
StatefulRouter
ContrastiveRouter
```

## Key Metrics to Compare

1. **Shakespeare**: Perplexity (↓), Character accuracy (↑)
2. **Math**: Exact string accuracy (↑)
3. **Expert Load Balance**: Imbalance ratio (→ 0 is perfect)
4. **Expert Specialization**: Which character/token types each expert handles

## Scaling Guide

| Config | d_model | layers | d_ff | params | Use case |
|--------|---------|--------|------|--------|----------|
| Small  | 64      | 2      | 128  | ~15K   | Smoke test |
| Medium | 128     | 3      | 256  | ~200K  | Quick dev |
| Large  | 256     | 4      | 512  | ~2-3M  | **Default** |
| XL     | 512     | 6      | 1024 | ~15M   | Final eval |

Override with CLI: `--d_model 512 --num_layers 6 --num_steps 5000`
