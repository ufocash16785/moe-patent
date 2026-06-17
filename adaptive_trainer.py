"""
=============================================================================
  Adaptive Bank Rebalancing Trainer
=============================================================================
  Purpose:  Train a banked MoE model where expert allocation per bank
            adapts automatically based on task difficulty.

  Starting state:
    Bank T (experts for Shakespeare): 4 experts  [0,1,2,3]
    Bank M (experts for Math):       4 experts  [4,5,6,7]

  Rebalancing rules:
    • Shakespeare too easy (acc≈1, loss<THRESHOLD_EASY):
        → move WORST expert from Bank T → Bank M
        → reset and retrain from scratch
        → Bank T shrinks, Bank M grows
        → stop when Bank T has MIN_T_EXPERTS (default=1)

    • Math too hard (math_acc=0, loss not converging):
        → move ONE expert from Bank M → Bank T
        → reset and retrain from scratch
        → Bank M shrinks, Bank T grows

  Inference: bank-constrained routing (top-1 bank determines top-2 bank)

  Usage:
    python adaptive_trainer.py                        # default settings
    python adaptive_trainer.py --init-t 4 --min-t 2   # start 4+4, stop at 2+6
    python adaptive_trainer.py --threshold-easy 0.0005
    python adaptive_trainer.py --dry-run               # smoke test, 10 steps only

  Author:  Cash Chou / Scripta
  Date:    2026-06-17
=============================================================================
"""

import argparse
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from benchmark_trainer import (
    make_large_config,
    download_shakespeare,
    TinyShakespeareDataset,
    MathDataset,
    compute_perplexity,
    compute_accuracy,
    compute_math_accuracy,
    BankedSparseMoEBlock,
)
from moe_model import MoETransformer


# ╔════════════════════════════════════════════════════════════════════════════╗
#  HYPERPARAMETERS
# ╚════════════════════════════════════════════════════════════════════════════╝

THRESHOLD_EASY_LOSS   = 1e-4   # Shakespeare loss below this → too easy
THRESHOLD_EASY_ACC    = 0.999  # Shakespeare acc above this → too easy
PLATEAU_WINDOW        = 2      # eval points to check for plateau (2 = check prev eval point)
PLATEAU_THRESHOLD     = 0.05   # plateau if loss improved < 5% over window
CONVERGENCE_WINDOW    = 3      # eval points for math oscillation check
DIVERGENCE_ACC_THRESH = 0.0    # Math acc below this → too hard

STEPS_PER_PHASE       = 1000   # training steps per phase before checking rebalance
EVAL_EVERY            = 100

# ╔════════════════════════════════════════════════════════════════════════════╗
#  HELPERS
# ╚════════════════════════════════════════════════════════════════════════════╝


def build_banked_model(config, bank_t_experts: list, bank_m_experts: list):
    """Build MoETransformer with BankedSparseMoEBlock using given bank config."""
    model = MoETransformer(config)
    for layer in model.layers:
        # Replace with BankedSparseMoEBlock using the current bank split
        layer.moe = BankedSparseMoEBlock(
            config,
            bank_t_experts=bank_t_experts,
            bank_m_experts=bank_m_experts,
        )
    return model


def set_all_banks(model, bank: Optional[str]):
    """Set active bank for all MoE layers."""
    for layer in model.layers:
        if hasattr(layer.moe, 'set_active_bank'):
            layer.moe.set_active_bank(bank)


def update_all_bank_config(model, bank_t_experts: list, bank_m_experts: list):
    """Propagate new bank expert assignments to all layers."""
    for layer in model.layers:
        layer.moe.update_bank_config(bank_t_experts, bank_m_experts)


def count_active_params(model, bank_t_experts: list, bank_m_experts: list) -> int:
    """Count total active parameters (all experts always trainable)."""
    return sum(p.numel() for p in model.parameters())


# ╔════════════════════════════════════════════════════════════════════════════╗
#  METRICS
# ╚════════════════════════════════════════════════════════════════════════════╝


def is_shakespeare_converged(metrics: Dict) -> bool:
    """
    Shakespeare has converged / plateaued if:
      (a) loss < THRESHOLD_EASY_LOSS AND acc > THRESHOLD_EASY_ACC,  OR
      (b) loss has NOT improved by more than PLATEAU_THRESHOLD over PLATEAU_WINDOW evals
    """
    losses = metrics.get("loss", [])
    accs   = metrics.get("accuracy", [])
    window = PLATEAU_WINDOW

    if len(losses) < 1:
        return False

    # Check (a): trivially easy
    recent_losses = losses[-min(window, len(losses)):]
    recent_accs   = accs[-min(window, len(accs)):]
    easy_loss = all(l < THRESHOLD_EASY_LOSS for l in recent_losses)
    easy_acc  = all(a > THRESHOLD_EASY_ACC  for a in recent_accs)
    if easy_loss and easy_acc:
        return True

    # Check (b): plateau (no significant loss improvement)
    if len(losses) >= window:
        window_losses = losses[-window:]
        oldest = window_losses[0]
        newest = window_losses[-1]
        if oldest > 0 and (oldest - newest) / oldest < PLATEAU_THRESHOLD:
            return True   # improved less than threshold

    return False


def is_math_too_hard(metrics: Dict) -> bool:
    """
    Check if Math accuracy is stuck at 0 (complete failure).
    """
    accs = metrics.get("accuracy", [])
    if len(accs) == 0:
        return False

    # Check last few steps — if all 0, it's too hard
    recent = accs[-min(5, len(accs)):]
    return all(a == 0.0 for a in recent)


def is_math_converged(metrics: Dict) -> bool:
    """
    Math has converged if:
      (a) acc stuck at 0 (complete failure),  OR
      (b) loss has NOT improved by more than PLATEAU_THRESHOLD over PLATEAU_WINDOW evals
    """
    losses = metrics.get("loss", [])
    accs   = metrics.get("accuracy", [])
    window = CONVERGENCE_WINDOW

    # (a) stuck at 0
    if len(accs) >= 3:
        recent_accs = accs[-min(3, len(accs)):]
        if all(a == 0.0 for a in recent_accs):
            return True

    # (b) plateau
    if len(losses) >= window:
        window_losses = losses[-window:]
        oldest = window_losses[0]
        newest = window_losses[-1]
        if oldest > 0 and (oldest - newest) / oldest < PLATEAU_THRESHOLD:
            return True

    return False


def find_worst_expert(model, dataset, config, bank_t_experts: list) -> int:
    """
    For the given bank, find the expert with the LOWEST total routing weight
    across a few batches (i.e., least-used / worst-performing expert).
    Returns the global expert index.
    """
    model.eval()
    expert_total_weights = {e: 0.0 for e in bank_t_experts}

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    data_iter = iter(loader)

    with torch.no_grad():
        for _ in range(min(10, len(dataset))):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                break
            x = x.to(config.device)

            hidden = model.token_embedding(x)
            if hidden.size(1) > config.max_seq_len:
                hidden = hidden[:, :config.max_seq_len, :]
            hidden = model.layers[0].ln2(hidden)

            _, expert_indices, _ = model.layers[0].moe.router(
                hidden, active_bank=None
            )
            # expert_indices: [M, 2]
            for e in bank_t_experts:
                count = (expert_indices == e).sum().item()
                expert_total_weights[e] += count

    worst_expert = min(expert_total_weights, key=expert_total_weights.get)
    model.train()
    return worst_expert


# ╔════════════════════════════════════════════════════════════════════════════╗
#  TRAINING LOOP (single phase)
# ╚════════════════════════════════════════════════════════════════════════════╝


def train_phase(
    model,
    config,
    dataset,
    bank_label: str,
    active_bank: str,
    steps: int = STEPS_PER_PHASE,
    is_math: bool = False,
    math_eval_dataset: Optional[MathDataset] = None,
    verbose: bool = True,
    convergence_checker=None,
) -> Tuple[Dict[str, List], bool]:
    """
    Train for one phase with the current bank config.

    Returns:
      (metrics_dict, converged_early):
        metrics_dict:  {'step':[], 'loss':[], 'accuracy':[], 'aux_loss':[]}
        converged_early: True if convergence detected during training (early return)
    """
    set_all_banks(model, active_bank)
    model.train()

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    data_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )

    metrics = {"step": [], "loss": [], "accuracy": [], "aux_loss": []}
    converged_early = False

    step = 0
    while step < steps:
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

        if step % EVAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                val_logits, val_aux = model(x)
                val_ce = F.cross_entropy(
                    val_logits.view(-1, config.vocab_size),
                    y.view(-1),
                    ignore_index=0,
                )
                accuracy = compute_accuracy(val_logits, y)

            metrics["step"].append(step)
            metrics["loss"].append(val_ce.item())
            metrics["accuracy"].append(accuracy)
            metrics["aux_loss"].append(val_aux.item())

            if verbose:
                tag = f"[{bank_label} Bank]"
                if is_math:
                    ma = compute_math_accuracy(model, math_eval_dataset, config, 50)
                    ppl = compute_perplexity(val_ce.item())
                    trend = ""
                    if len(metrics["loss"]) >= 2:
                        prev = metrics["loss"][-2]
                        curr = val_ce.item()
                        if curr > prev * 1.01:
                            trend = "  ⚠️ worsening"
                        elif curr < prev * 0.99:
                            trend = "  ↓ improving"
                    print(f"  {tag} Step {step:5d} | Loss: {val_ce.item():.4f} | "
                          f"CharAcc: {accuracy:.4f} | MathAcc: {ma:.4f} | PPL: {ppl:.2f}{trend}")
                else:
                    ppl = compute_perplexity(val_ce.item())
                    print(f"  {tag} Step {step:5d} | Loss: {val_ce.item():.4f} | "
                          f"Acc: {accuracy:.4f} | PPL: {ppl:.2f}")

            # ── Early convergence check ────────────────────────────────────
            if convergence_checker is not None and convergence_checker(metrics):
                converged_early = True
                if verbose:
                    print(f"  ⚡ Early convergence at step {step} — stopping phase")
                break

    return metrics, converged_early


# ╔════════════════════════════════════════════════════════════════════════════╗
#  MAIN ADAPTIVE LOOP
# ╚════════════════════════════════════════════════════════════════════════════╝


def run_adaptive_trainer(
    config=None,
    init_t_experts: int = 4,
    min_t_experts: int = 1,
    shakespeare_steps: int = STEPS_PER_PHASE,
    math_steps: int = STEPS_PER_PHASE,
    verbose: bool = True,
) -> Tuple[MoETransformer, List[Dict]]:
    """
    Run the adaptive bank rebalancing loop.

    Returns:
      model: final trained model
      history: list of per-phase metrics dicts
    """
    print("\n" + "="*65)
    print("  Adaptive Bank Rebalancing Trainer")
    print("="*65)

    if config is None:
        config = make_large_config(max_seq_len=128)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Datasets ─────────────────────────────────────────────────────────
    print("\n[Data]")
    shakespeare_data = download_shakespeare()
    shakespeare_dataset = TinyShakespeareDataset(shakespeare_data, config.max_seq_len, stride=64)
    math_train = MathDataset(10000, config.max_seq_len, difficulty="easy")
    math_eval  = MathDataset(500,   config.max_seq_len, difficulty="easy")
    print(f"  Shakespeare: {len(shakespeare_dataset):,} samples")
    print(f"  Math train:  {len(math_train):,} | eval: {len(math_eval):,}")

    # ── Initial bank config ──────────────────────────────────────────────
    all_experts = list(range(8))
    bank_t_experts = all_experts[:init_t_experts]          # e.g. [0,1,2,3]
    bank_m_experts = all_experts[init_t_experts:]           # e.g. [4,5,6,7]

    # Build model
    model = build_banked_model(config, bank_t_experts, bank_m_experts).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params: {total_params:,}")
    print(f"  Initial: Bank T = {bank_t_experts} ({len(bank_t_experts)} experts)")
    print(f"           Bank M = {bank_m_experts} ({len(bank_m_experts)} experts)")

    history = []
    phase_num = 0

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        phase_num += 1
        t_size = len(bank_t_experts)
        m_size = len(bank_m_experts)
        print(f"\n{'='*60}")
        print(f"  Phase {phase_num}  |  Bank T: {t_size} experts  |  Bank M: {m_size} experts")
        print(f"{'='*60}")

        # ── Phase 1: Train Shakespeare ─────────────────────────────────
        print(f"\n[Phase {phase_num} — Shakespeare | Bank T active]")
        print(f"  Current banks → Bank T: {bank_t_experts}  |  Bank M: {bank_m_experts}")
        sh_metrics, sh_early = train_phase(
            model, config, shakespeare_dataset,
            bank_label='T', active_bank='T',
            steps=shakespeare_steps,
            is_math=False,
            verbose=verbose,
            convergence_checker=is_shakespeare_converged,
        )
        final_sh_loss = sh_metrics["loss"][-1]
        final_sh_acc  = sh_metrics["accuracy"][-1]
        history.append({"phase": phase_num, "task": "shakespeare",
                         "bank_t": list(bank_t_experts), "bank_m": list(bank_m_experts),
                         "metrics": sh_metrics})

        print(f"\n  Shakespeare final → Loss: {final_sh_loss:.4f} | Acc: {final_sh_acc:.4f}")

        # ── Check: Shakespeare converged? ──────────────────────────────
        sh_converged = is_shakespeare_converged(sh_metrics)

        if sh_converged and t_size > min_t_experts:
            worst = find_worst_expert(model, shakespeare_dataset, config, bank_t_experts)
            print(f"\n  ⚠️  Shakespeare converged! (loss={final_sh_loss:.6f})")
            print(f"  → Moving worst expert {worst} from Bank T → Bank M")

            bank_t_experts = [e for e in bank_t_experts if e != worst]
            bank_m_experts = sorted(bank_m_experts + [worst])
            update_all_bank_config(model, bank_t_experts, bank_m_experts)
            print(f"  New: Bank T = {bank_t_experts}  |  Bank M = {bank_m_experts}")
            continue  # reset and retrain

        elif sh_converged and t_size <= min_t_experts:
            print(f"\n  Shakespeare converged but Bank T at minimum ({min_t_experts}) — skip rebalance")

        # ── Phase 2: Train Math ────────────────────────────────────────
        print(f"\n[Phase {phase_num} — Math | Bank M active]")
        print(f"  Current banks → Bank T: {bank_t_experts}  |  Bank M: {bank_m_experts}")
        math_metrics, math_early = train_phase(
            model, config, math_train,
            bank_label='M', active_bank='M',
            steps=math_steps,
            is_math=True,
            math_eval_dataset=math_eval,
            verbose=verbose,
            convergence_checker=is_math_converged,
        )
        final_math_acc = math_metrics["accuracy"][-1]
        final_math_loss = math_metrics["loss"][-1]
        history.append({"phase": phase_num, "task": "math",
                        "bank_t": list(bank_t_experts), "bank_m": list(bank_m_experts),
                        "metrics": math_metrics})

        print(f"\n  Math final → Loss: {final_math_loss:.4f} | Acc: {final_math_acc:.4f}")

        # ── Check: Math converged? ─────────────────────────────────────
        math_converged = is_math_converged(math_metrics)

        if math_converged and len(bank_t_experts) > 1:
            print(f"\n  ⚠️  Math converged (plateau/acc=0)!")
            print(f"  → Moving worst Bank T expert → Bank M  (strengthen Bank M for math)")
            worst_t = find_worst_expert(model, shakespeare_dataset, config, bank_t_experts)
            bank_t_experts = [e for e in bank_t_experts if e != worst_t]
            bank_m_experts = sorted(bank_m_experts + [worst_t])
            update_all_bank_config(model, bank_t_experts, bank_m_experts)
            print(f"  New: Bank T = {bank_t_experts}  |  Bank M = {bank_m_experts}")
            continue  # reset and retrain
        elif math_converged and len(bank_t_experts) <= 1:
            print(f"\n  Math converged but Bank T at minimum — cannot rebalance T→M")

        # ── Check: Both stable (no rebalance needed)? ───────────────────
        if not sh_converged and not math_converged:
            print(f"\n  ✅ Both tasks still improving. Stopping adaptive loop.")
            break

        # Safety: prevent infinite loop
        if phase_num >= 5:
            print(f"\n  Safety limit (5 phases) reached. Stopping.")
            break

    # ── Final inference evaluation ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Final Model  |  Bank T: {bank_t_experts}  |  Bank M: {bank_m_experts}")
    print(f"{'='*60}")

    set_all_banks(model, None)  # inference mode
    model.eval()

    with torch.no_grad():
        x_sh, y_sh = shakespeare_dataset[0]
        x_sh = x_sh.unsqueeze(0).to(config.device)
        y_sh = y_sh.to(config.device)
        logits_sh, _ = model(x_sh)
        ppl_sh = compute_perplexity(
            F.cross_entropy(logits_sh.view(-1, 256), y_sh.view(-1), ignore_index=0).item()
        )

        math_acc_final = compute_math_accuracy(model, math_eval, config, 500)

    print(f"\n  Shakespeare PPL (inference): {ppl_sh:.2f}")
    print(f"  Math Acc (inference):         {math_acc_final:.4f}")

    # ── Expert utilisation report ───────────────────────────────────────
    expert_counts = torch.zeros(8)
    for i in range(50):
        x, _ = shakespeare_dataset[i % len(shakespeare_dataset)]
        x = x.unsqueeze(0).to(config.device)
        hidden = model.token_embedding(x) + model.pos_embedding[:, :x.size(1), :]
        hidden = model.layers[0].ln2(hidden)
        with torch.no_grad():
            _, idx, _ = model.layers[0].moe.router(hidden, active_bank=None)
        for e in range(8):
            expert_counts[e] += (idx == e).sum().item()

    print(f"\n  Expert utilisation (top-1, inference):")
    print(f"    Bank T: {bank_t_experts}")
    print(f"    Bank M: {bank_m_experts}")
    for e in range(8):
        marker = " ◀ Bank T" if e in bank_t_experts else " ◀ Bank M"
        print(f"      Expert {e}: {int(expert_counts[e]):>5}{marker}")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ADAPTIVE TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Phase':>5}  {'Task':>12}  {'Bank T':>12}  {'Bank M':>12}  {'Sh PPL':>8}  {'Math Acc':>8}")
    print(f"  {'-'*60}")
    for h in history:
        sh_ppl = compute_perplexity(h['metrics']['loss'][-1]) if h['task'] == 'shakespeare' else '—'
        ma = h['metrics']['accuracy'][-1] if h['task'] == 'math' else '—'
        sh_str = f"{sh_ppl:.2f}" if isinstance(sh_ppl, float) else sh_ppl
        ma_str = f"{ma:.4f}" if isinstance(ma, float) else ma
        print(f"  {h['phase']:>5}  {h['task']:>12}  {str(h['bank_t']):>12}  "
              f"{str(h['bank_m']):>12}  {sh_str:>8}  {ma_str:>8}")

    return model, history


# ╔════════════════════════════════════════════════════════════════════════════╗
#  CLI
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Bank Rebalancing Trainer")
    parser.add_argument("--init-t", type=int, default=4,
                        help="Initial number of experts in Bank T (default: 4)")
    parser.add_argument("--min-t", type=int, default=1,
                        help="Minimum experts in Bank T before stopping (default: 1)")
    parser.add_argument("--threshold-easy", type=float, default=None,
                        help="Override Shakespeare easy-loss threshold")
    parser.add_argument("--shakespeare-steps", type=int, default=STEPS_PER_PHASE,
                        help=f"Steps per Shakespeare phase (default: {STEPS_PER_PHASE})")
    parser.add_argument("--math-steps", type=int, default=STEPS_PER_PHASE,
                        help=f"Steps per Math phase (default: {STEPS_PER_PHASE})")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 30 steps only")
    args = parser.parse_args()

    if args.threshold_easy is not None:
        THRESHOLD_EASY_LOSS = args.threshold_easy

    if args.dry_run:
        print("⚠️  DRY RUN — using 30 steps per phase")
        ds, ms = 30, 30
    else:
        ds, ms = args.shakespeare_steps, args.math_steps

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = make_large_config(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_steps=max(ds, ms),
        batch_size=args.batch_size,
    )
    config.device = device

    model, history = run_adaptive_trainer(
        config=config,
        init_t_experts=args.init_t,
        min_t_experts=args.min_t,
        shakespeare_steps=ds,
        math_steps=ms,
        verbose=True,
    )

    print("\n✅ Adaptive training complete")
