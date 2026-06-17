"""
=============================================================================
  Router Variants — Ideas You Can Swap Into moe_model.py
=============================================================================
  Each class in this file follows the same interface as MoERouter:
      forward(hidden_states) -> (weights, expert_indices, router_logits)

  Swap one in by changing SparseMoEBlock.__init__:
      self.router = RouterVariant(config)

  💡 MANY of these are patent-eligible ideas.  Some are known in the
     literature; others may be novel.  The comments note where each fits.
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT A — Sigmoid + Threshold Routing                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  Instead of enforcing a fixed top‑k, apply sigmoid to logits and route to
#  every expert whose probability exceeds a threshold.
#  - Dynamic k per token
#  - Could route to 0 or all experts
#  PATENT ANGLE: adaptive compute budget per token

class SigmoidThresholdRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.threshold = 0.3  # Tunable hyper‑parameter

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        logits = self.gate(hidden_states)
        probs = torch.sigmoid(logits)

        # ── Mask: experts above threshold ──
        mask = probs > self.threshold
        # Ensure at least one expert per token (fallback to highest)
        has_none = ~mask.any(dim=-1)
        if has_none.any():
            fallback = probs[has_none].argmax(dim=-1)
            mask[has_none] = F.one_hot(fallback, self.num_experts).bool()

        # ── Construct top‑k interface ──
        # We need to return (weights, indices, logits) like the standard router.
        # Map dynamic selection into a fixed top_k = num_experts output,
        # with zero weights for unselected experts.
        routing_weights = probs * mask.float()
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Sort so selected experts appear first
        weights_sorted, indices_sorted = routing_weights.sort(dim=-1, descending=True)

        # Standard router returns top_k — here we use num_experts as an
        # upper bound, but only the non‑zero ones are "selected".
        # ⚠️ You need to modify SparseMoEBlock to handle dynamic k.
        return weights_sorted, indices_sorted, logits.unsqueeze(0) if logits.dim() == 2 else logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT B — Hierarchical Router (cluster first, then expert)             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  Experts are grouped into G clusters of (num_experts / G) experts each.
#  Step 1: route token to top‑2 clusters.
#  Step 2: within each selected cluster, route to top‑1 expert.
#  - Reduces routing compute: G-way vs E-way top‑k
#  PATENT ANGLE: hierarchical / coarse‑to‑fine routing for scalable MoE

class HierarchicalRouter(nn.Module):
    def __init__(self, config, num_clusters: int = 4):
        super().__init__()
        assert config.num_experts % num_clusters == 0
        self.num_clusters = num_clusters
        self.experts_per_cluster = config.num_experts // num_clusters
        self.num_experts = config.num_experts

        self.cluster_gate = nn.Linear(config.d_model, num_clusters, bias=False)
        self.expert_gate = nn.Linear(config.d_model, config.num_experts, bias=False)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        # Step 1: cluster selection (top‑2 out of G)
        cluster_logits = self.cluster_gate(hidden_states)           # [M, G]
        cluster_probs = F.softmax(cluster_logits, dim=-1)
        cluster_w, cluster_idx = torch.topk(cluster_probs, 2, dim=-1)  # [M, 2]

        # Step 2: expert selection within each cluster
        # We compute all expert logits but only keep those in the chosen clusters.
        expert_logits = self.expert_gate(hidden_states)  # [M, E]
        expert_probs = F.softmax(expert_logits, dim=-1, dtype=torch.float32)  # [M, E]

        # ── Mask: zero out experts not in the selected clusters ──
        # Build a cluster->expert mapping vector
        cluster_to_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )
        for c in range(self.num_clusters):
            start = c * self.experts_per_cluster
            end = start + self.experts_per_cluster
            cluster_to_expert[start:end] = c  # [E]

        # For each token: which clusters were selected? cluster_idx: [M, 2]
        # Build a boolean mask [M, E]: True if the expert's cluster is in
        # the top-2 clusters for this token.
        c0 = cluster_idx[:, 0]  # [M]
        c1 = cluster_idx[:, 1]  # [M]
        mask0 = (cluster_to_expert.unsqueeze(0) == c0.unsqueeze(-1))  # [M, E]
        mask1 = (cluster_to_expert.unsqueeze(0) == c1.unsqueeze(-1))
        combined_mask = (mask0 | mask1).float()

        # Apply mask and re-normalise so probabilities sum to 1
        masked_probs = expert_probs * combined_mask
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Top-2 among allowed experts
        top_w, top_idx = torch.topk(masked_probs, 2, dim=-1)

        return top_w, top_idx, expert_logits.unsqueeze(0) if expert_logits.dim() == 2 else expert_logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT C — Content‑Adaptive Top‑k (entropy‑based)                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  For each token, compute softmax entropy over experts. Low entropy = the
#  router is confident → use fewer experts (e.g., k=1). High entropy = the
#  token is ambiguous → use more experts (e.g., k=4).
#  PATENT ANGLE: dynamic, per‑token expert count based on routing confidence
#  REFERENCES: No widely‑known paper uses this exact approach.

class EntropyAdaptiveRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.base_k = config.top_k  # e.g., 2

        # Learnable thresholds for low/high entropy
        self.low_threshold = nn.Parameter(torch.tensor(0.5))
        self.high_threshold = nn.Parameter(torch.tensor(1.5))

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        logits = self.gate(hidden_states)              # [M, E]
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        # Compute entropy per token: H(p) = -Σ p_i log p_i
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # [M]

        # Determine k per token (clamped to [1, num_experts])
        # Low entropy → confident → fewer experts
        # High entropy → uncertain → more experts
        k_low = max(1, self.base_k - 1)
        k_high = min(self.num_experts, self.base_k + 2)

        k_values = torch.full_like(entropy, self.base_k, dtype=torch.long)
        k_values[entropy < self.low_threshold] = k_low
        k_values[entropy > self.high_threshold] = k_high

        # Select top‑k per token where k varies
        top_w_list = []
        top_idx_list = []

        for i, k in enumerate(k_values.tolist()):
            w, idx = torch.topk(probs[i:i+1], min(k, self.num_experts), dim=-1)
            top_w_list.append(w)
            top_idx_list.append(idx)

        # Pad to max k for a fixed output shape
        max_k = max(k_high, self.base_k)
        padded_weights = torch.zeros(len(k_values), max_k, device=probs.device)
        padded_indices = torch.zeros(len(k_values), max_k, dtype=torch.long, device=probs.device)

        for i, (w, idx) in enumerate(zip(top_w_list, top_idx_list)):
            n = w.shape[-1]
            padded_weights[i, :n] = w
            padded_indices[i, :n] = idx
            if n > 0:
                padded_weights[i, :n] /= padded_weights[i, :n].sum() + 1e-9

        return padded_weights, padded_indices, logits.unsqueeze(0) if logits.dim() == 2 else logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT D — Expert Embedding / Attention‑Based Routing                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  Instead of a linear projection W_g · x, each expert has a learnable
#  "embedding" vector.  The router computes attention scores between the
#  token hidden state and each expert embedding.
#  PATENT ANGLE: content‑expert attention instead of linear gating
#  REFERENCES: Resembles "soft MoE" / "MoC LE" but with learned expert keys.

class AttentionRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_model = config.d_model

        # Learnable expert embeddings (like "keys" in attention)
        self.expert_embeddings = nn.Parameter(
            torch.randn(config.num_experts, config.d_model) * 0.02
        )
        # Value projection for experts (optional)
        # self.expert_values = nn.Parameter(...)

        # Optional: learned temperature
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        # [M, D]

        # ── Attention scores: token · expert_embedding ──
        # c.f. dot‑product attention: Q = x, K = expert_embeddings
        logits = hidden_states @ self.expert_embeddings.T  # [M, E]
        logits = logits * self.logit_scale.exp()           # learned temperature

        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)
        top_w = top_w.to(hidden_states.dtype)

        return top_w, top_idx, logits.unsqueeze(0) if logits.dim() == 2 else logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT E — Gumbel‑Softmax Router (stochastic during training)           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  Uses Gumbel‑softmax reparameterisation for differentiable discrete routing.
#  - During training: stochastic (Gumbel noise added before softmax)
#  - During inference: deterministic (standard softmax + top‑k)
#  PATENT ANGLE: exploration in routing via Gumbel noise schedule
#  REFERENCES: Used in NLLB‑MoE (second expert); Phi‑MoE uses similar idea.

class GumbelRouter(nn.Module):
    def __init__(self, config, temperature: float = 1.0):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.temperature = temperature

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        logits = self.gate(hidden_states)  # [M, E]

        if self.training:
            # Add Gumbel noise
            uniform = torch.rand_like(logits)
            gumbel = -(-uniform.log()).log()
            logits = logits + gumbel * self.temperature

        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)
        top_w = top_w.to(hidden_states.dtype)

        return top_w, top_idx, logits.unsqueeze(0) if logits.dim() == 2 else logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT F — Residual / Skip‑Connection Router                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  The router gets input from BOTH the current layer's hidden state AND
#  a "residual" state from the previous MoE layer's routing decisions.
#  PATENT ANGLE: routing with memory / temporal consistency
#  IDEA: tokens that were routed to the same expert in layer L‑1 get a
#        bonus to the same expert in layer L.

class StatefulRouter(nn.Module):
    """
    Simple stateful router: keeps a running hidden state that captures
    which experts were used for the previous token position.
    """
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model + config.num_experts, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.prev_state = None

    def reset_state(self):
        self.prev_state = None

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        B, T = orig_shape[0], orig_shape[1]
        if hidden_states.dim() == 3:
            hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
        else:
            hidden_states_flat = hidden_states

        # Build state: previous routing decisions or zeros
        if self.prev_state is None:
            state = torch.zeros(B, T, self.num_experts, device=hidden_states.device)
        else:
            # Shift state: copy from prev timestep
            state = F.pad(self.prev_state[:, :-1, :], (0, 0, 1, 0))

        state_flat = state.view(-1, self.num_experts)
        combined = torch.cat([hidden_states_flat, state_flat], dim=-1)

        logits = self.gate(combined)
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)
        top_w = top_w.to(hidden_states.dtype)

        # Save routing decision as state for next forward pass
        self.prev_state = F.one_hot(top_idx, self.num_experts).float().sum(dim=-2)

        if hidden_states.dim() == 3:
            logits_out = logits.view(B, T, -1)
        else:
            logits_out = logits

        return top_w, top_idx, logits_out


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT G — Experts as "Retrieved Memory" (LoRA‑style)                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  Instead of having full MLP experts, each "expert" is a low‑rank adapter
#  (LoRA) that modifies a shared base MLP.  The router selects which adapters
#  to apply.
#  PATENT ANGLE: MoE as dynamic model editing / retrieval

class LoRAMoERouter(nn.Module):
    """
    Each expert is a pair of low‑rank matrices (A_i, B_i) applied as:
        W_shared(x) + Σ_i w_i * B_i(A_i(x))
    The router selects which adapters to activate.
    """
    def __init__(self, config, rank: int = 8):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.rank = rank

        # LoRA adapters per expert: A: [d_model, rank], B: [rank, d_model]
        self.lora_a = nn.Parameter(
            torch.randn(config.num_experts, config.d_model, rank) * 0.02
        )
        self.lora_b = nn.Parameter(
            torch.randn(config.num_experts, rank, config.d_model) * 0.02
        )

    def get_adapter_output(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Return B_i(A_i(x))"""
        a = self.lora_a[expert_idx]  # [D, r]
        b = self.lora_b[expert_idx]  # [r, D]
        return x @ a @ b

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
        else:
            hidden_states_flat = hidden_states

        logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)
        top_w = top_w.to(hidden_states.dtype)

        if hidden_states.dim() == 3:
            logits_out = logits.view(orig_shape[0], orig_shape[1], -1)
        else:
            logits_out = logits

        return top_w, top_idx, logits_out


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  VARIANT H — Contrastive Router (maximises expert diversity)              ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#  The router is trained with an additional loss that maximises the
#  divergence between expert outputs or routing distributions.
#  PATENT ANGLE: explicit diversity maximisation in expert specialisation

class ContrastiveRouter(nn.Module):
    """
    Standard router + additional contrastive loss.
    The contrastive loss is computed externally (in the training loop).
    This variant adds a projection head for computing contrastive features.
    """
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        # Projection head for contrastive loss
        self.projection = nn.Linear(config.d_model, 32, bias=False)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
        else:
            hidden_states_flat = hidden_states

        logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)
        top_w = top_w.to(hidden_states.dtype)

        if hidden_states.dim() == 3:
            logits_out = logits.view(orig_shape[0], orig_shape[1], -1)
        else:
            logits_out = logits

        return top_w, top_idx, logits_out


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TEST: Run the standard model with a swapped‑in variant                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Quick smoke test — swap any router variant into the MoE model
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__) or ".")
    from moe_model import (
        MoEConfig, MoERouter, ExpertMLP, SparseMoEBlock, MoETransformer,
    )

    config = MoEConfig(
        num_experts=4, top_k=2, d_model=32, d_ff=64,
        num_layers=1, num_steps=50, num_heads=2, max_seq_len=16,
    )

    print("Testing router variants...\n")

    variants = [
        ("Standard (no change)", None),
        ("AttentionRouter", AttentionRouter),
        ("EntropyAdaptive", EntropyAdaptiveRouter),
        ("GumbelRouter", GumbelRouter),
        ("SigmoidThreshold", SigmoidThresholdRouter),
        ("HierarchicalRouter", HierarchicalRouter),
        ("ContrastiveRouter", ContrastiveRouter),
    ]

    for name, RouterCls in variants:
        # Swap the router class via the class attribute
        orig_router_cls = SparseMoEBlock.router_class

        if RouterCls is not None:
            SparseMoEBlock.router_class = RouterCls
        else:
            SparseMoEBlock.router_class = MoERouter

        try:
            model = MoETransformer(config)
            x = torch.randint(0, config.vocab_size, (2, 8))
            logits, aux_loss = model(x)
            print(f"  [OK] {name:25s}  output {tuple(logits.shape)}, aux_loss={aux_loss.item():.4f}")
        except Exception as e:
            print(f"  [FAIL] {name:25s}  FAILED: {e}")
            import traceback
            traceback.print_exc()

        SparseMoEBlock.router_class = orig_router_cls
