"""
=============================================================================
  Mixture of Experts (MoE) Language Model — Minimal Reference Implementation
=============================================================================
  Purpose:  A small, self-contained MoE decoder-only transformer that you can
            run on CPU.  The router is isolated and heavily commented so you
            can modify it, try new ideas, and verify they work end-to-end.

  Design choices:
    - Top-2 routing (standard for MoE LMs like Mixtral 8x7B, OLMoE, DBRX).
    - 8 experts, 2 selected per token.
    - Load-balancing ("auxiliary") loss to keep expert usage balanced.
    - Full training loop on a tiny synthetic task: copy a sequence token-by-token.
    - ~15K parameters — runs in seconds on CPU.

  Files in this directory:
    moe_model.py      — This file: model definition + training + inference.
    router_variants.py — (optional) Alternative router designs you can swap in.

  Author:  [Your Name]
  Date:    2026-06-16
=============================================================================
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class MoEConfig:
    """Hyper‑parameters for our mini MoE transformer."""

    def __init__(self, **kwargs):
        # ── Vocabulary & embeddings ──────────────────────────────────
        self.vocab_size: int = 256       # Tiny vocabulary (byte-level)
        self.d_model: int = 64           # Hidden dimension

        # ── Transformer layers ───────────────────────────────────────
        self.num_layers: int = 2         # Number of decoder layers
        self.num_heads: int = 4          # Attention heads (d_model must be divisible)
        self.d_ff: int = 128             # Intermediate dimension in FFN experts

        # ── MoE specifics ────────────────────────────────────────────
        self.num_experts: int = 8        # Total experts
        self.top_k: int = 2              # Experts selected per token
        self.router_aux_loss_coef: float = 0.01  # Weight for load-balancing loss

        # ── Training ─────────────────────────────────────────────────
        self.max_seq_len: int = 32
        self.dropout: float = 0.1
        self.learning_rate: float = 1e-3
        self.batch_size: int = 8
        self.num_steps: int = 200

        # ── Device ───────────────────────────────────────────────────
        self.device: str = "cpu"

        # Override any with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER — THE CORE OF THE MOE LAYER                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
#  Every MoE paper introduces a novel routing mechanism.  This is the
#  "vanilla" top‑2 router — the starting point you should modify.
#
#  💡 IDEAS TO TRY (patent angles):
#     1.  Sigmoid + threshold instead of softmax + top‑k
#     2.  Hierarchical routing — cluster experts, pick cluster then expert
#     3.  Learnable temperature per expert
#     4.  Router that looks at the *residual stream* not the MLP input
#     5.  Expert "affinity" matrix (embedding per expert, dot‑product with token)
#     6.  Context‑dependent top‑k (different k per token based on entropy)
#     7.  State‑ful router with learned hidden state (memory across tokens)
#     8.  Two‑stage: coarse router (which group) then fine router (which expert)
#     9.  Router with auxiliary prediction heads (predict task difficulty)
#    10.  Contrastive router (maximizes inter‑expert divergence)
#    11.  Grouped experts that cannot be selected together
#    12.  Learnable expert embeddings + attention-based routing

class MoERouter(nn.Module):
    """
    Standard top‑k router (used by Mixtral, OLMoE, DBRX, JetMoE, etc.)
    ───────────────────────────────────────────────────────────────────
    Forward pass:
      1.  hidden_states  →  Linear(d_model, num_experts)  →  logits
      2.  softmax over experts  →  routing probabilities
      3.  top‑k selection      →  (weights, expert_indices)
      4.  re‑normalise the k selected weights (they sum to 1)

    Returns:
      routing_weights  — [batch*seq, top_k]     (float, sum=1 per token)
      selected_experts — [batch*seq, top_k]     (long, expert IDs per token)
      router_logits    — [batch*seq, num_experts] (float, used for aux loss)
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # ── The only learned parameter of the router ─────────────────
        # Shape: [d_model, num_experts]
        # A linear projection from the hidden space into "expert score" space.
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)

        # Optional: learnable temperature per expert (idea starter)
        # self.log_temperature = nn.Parameter(torch.zeros(self.num_experts))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          hidden_states: [batch_size, seq_len, d_model]  (or flattened)
        Returns:
          routing_weights, selected_experts, router_logits
        """
        orig_shape = hidden_states.shape
        # Flatten batch & sequence dimensions
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        # Now: [batch * seq, d_model]

        # ── Step 1: Compute raw expert scores ───────────────────────
        #         scores = x · W_gate     [M, E]
        router_logits = self.gate(hidden_states)          # [M, num_experts]

        # ── Step 2: Convert scores to probabilities ─────────────────
        # Softmax over the expert dimension.
        # NOTE: Cast to float32 for numerical stability (common trick).
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # ── Step 3: Select top‑k experts per token ──────────────────
        # topk returns (values, indices) both shaped [M, top_k]
        top_weights, top_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # ── Step 4: Re-normalise so the k weights sum to 1 ──────────
        # (This is standard in Mixtral and most implementations.)
        top_weights = top_weights / (
            top_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Cast back to original dtype
        top_weights = top_weights.to(hidden_states.dtype)

        # Reshape to match input if it was 3D
        if len(orig_shape) == 3:
            router_logits = router_logits.view(orig_shape[0], orig_shape[1], -1)
        # router_logits stays as the raw logits (before softmax) for aux loss

        return top_weights, top_experts, router_logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  EXPERT MLP                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class ExpertMLP(nn.Module):
    """A single feed‑forward expert:  Linear → ReLU → Linear (standard FFN)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SPARSE MOE BLOCK                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
#  Combines the router + N experts.  This block replaces the standard FFN
#  in each transformer layer.

class SparseMoEBlock(nn.Module):
    """
    A complete MoE layer:
      1. Router assigns each token to top‑k experts
      2. Each selected expert processes the token
      3. Outputs are weighted by router probabilities and summed

    Expert dispatch (which token → which expert) uses the standard
    one‑hot + mask + loop strategy from the Mixtral / HuggingFace codebase.
    """
    # ── Class attribute: override to swap in alternative routers ──
    # Usage:
    #   SparseMoEBlock.router_class = AttentionRouter
    # OR subclass and set:
    #   class MyMoEBlock(SparseMoEBlock):
    #       router_class = MyRouter
    router_class = MoERouter

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_model = config.d_model

        # Router — uses the class attribute, so you can swap router_class
        self.router = self.router_class(config)

        # Expert MLPs — stored in a ModuleList so PyTorch tracks them
        self.experts = nn.ModuleList([
            ExpertMLP(config.d_model, config.d_ff)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [batch, seq, d_model]
        Returns:
          final_output: [batch, seq, d_model] — weighted sum of expert outputs
          router_logits: [batch, seq, num_experts] — for aux loss computation
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # ── Route ────────────────────────────────────────────────────
        weights, expert_indices, router_logits = self.router(x)
        # weights:       [num_tokens, top_k]
        # expert_indices:[num_tokens, top_k]
        # router_logits: [batch, seq, num_experts]

        # ── Dispatch tokens to experts ───────────────────────────────
        # Standard approach: one‑hot encode → compute per‑expert masks.
        # Shape: [num_tokens, top_k, num_experts]
        expert_mask = F.one_hot(
            expert_indices, num_classes=self.num_experts
        ).float()

        # Aggregate: for each expert, which tokens (and which of top_k slots)
        # are assigned to it?  Shape: [num_experts, num_tokens, top_k]
        expert_mask = expert_mask.permute(2, 0, 1)  # [E, M, k]

        # ═══  Per‑expert computation  ═══════════════════════════════
        # We loop over experts to keep this readable.
        # (Production code uses fused kernels: vLLM's fused_moe, Megablocks, etc.)
        final_output = torch.zeros(
            num_tokens, d_model, device=x.device, dtype=x.dtype
        )

        for expert_idx in range(self.num_experts):
            # Which tokens chose this expert, and in which top‑k slot?
            # mask: [num_tokens, top_k] — binary for this expert
            mask = expert_mask[expert_idx]  # [M, k]

            # Sum over top‑k slots → per‑token assignment: 0 or 1 (or more if
            # an expert is selected in multiple top‑k slots — rare but possible)
            token_assignment = mask.sum(dim=-1)  # [M]
            selected = token_assignment > 0

            if not selected.any():
                continue

            # Selected token indices
            selected_indices = torch.where(selected)[0]
            expert_input = x.view(num_tokens, d_model)[selected_indices]

            # Run through the expert MLP
            expert_output = self.experts[expert_idx](expert_input)

            # Weight by router probability
            # For each selected token, sum weights across top‑k slots that
            # pointed to this expert
            expert_weight = (weights * mask).sum(dim=-1)  # [M]
            expert_weight = expert_weight[selected_indices]

            # Accumulate
            final_output[selected_indices] += (
                expert_output * expert_weight.unsqueeze(-1)
            )

        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output, router_logits

    @staticmethod
    def compute_auxiliary_loss(
        router_logits: torch.Tensor, num_experts: int, top_k: int
    ) -> torch.Tensor:
        """
        Load‑balancing (auxiliary) loss from Switch Transformers.

        L_aux = num_experts * Σ_i (f_i · P_i)

        where:
          f_i = fraction of tokens routed to expert i
          P_i = average routing probability assigned to expert i

        Minimised when all experts have equal f_i and P_i (= 1/num_experts).
        """
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # [batch*seq, num_experts]

        _, selected = torch.topk(routing_weights, top_k, dim=-1)
        # [batch*seq, top_k]

        expert_mask = F.one_hot(selected, num_experts).float()
        # [batch*seq, top_k, num_experts]

        # f_i: fraction of (token, slot) pairs assigned to expert i
        f_i = expert_mask.mean(dim=(0, 1))  # [num_experts]

        # P_i: average probability assigned to expert i across all tokens
        P_i = routing_weights.mean(dim=0)    # [num_experts]

        loss = (f_i * P_i).sum() * num_experts
        return loss


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SINGLE‑HEAD ATTENTION (for simplicity)                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class CausalSelfAttention(nn.Module):
    """Simple multi‑head causal attention."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: [B, T, nh, hd]

        # Move head dim to position 1 for standard matmul: [B, nh, T, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # causal mask [T, T]
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        att = att + mask.unsqueeze(0).unsqueeze(0)                  # broadcast
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v                                                  # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.proj(y)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DECODER LAYER  (Attention + MoE + Residuals + LayerNorm)                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class MoEDecoderLayer(nn.Module):
    """
    Standard pre‑norm transformer decoder layer where the FFN is
    replaced by a SparseMoEBlock.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = SparseMoEBlock(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention sub‑layer
        x = x + self.attention(self.ln1(x))
        # MoE sub‑layer (replaces MLP)
        moe_out, router_logits = self.moe(self.ln2(x))
        x = x + moe_out
        return x, router_logits


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FULL MODEL                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class MoETransformer(nn.Module):
    """
    Decoder‑only transformer with MoE in every layer.

    The total loss is:
        L = cross_entropy(next_token_prediction)
          + aux_loss_coef * Σ(layer_aux_loss)
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            MoEDecoderLayer(config) for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie embeddings
        self.token_embedding.weight = self.lm_head.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          input_ids: [batch, seq]
        Returns:
          logits:     [batch, seq, vocab_size]
          aux_loss:   scalar (sum of all MoE layer aux losses)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len

        # Token + position embeddings
        tok = self.token_embedding(input_ids)         # [B, T, D]
        pos = self.pos_embedding[:, :T, :]            # [1, T, D]
        x = self.dropout(tok + pos)

        aux_loss = 0.0
        for layer in self.layers:
            x, router_logits = layer(x)
            aux_loss = aux_loss + SparseMoEBlock.compute_auxiliary_loss(
                router_logits.view(-1, self.config.num_experts),
                self.config.num_experts,
                self.config.top_k,
            )

        x = self.ln_f(x)
        logits = self.lm_head(x)                      # [B, T, V]
        return logits, aux_loss

    def generate(self, prompt: torch.Tensor, max_new: int = 16) -> torch.Tensor:
        """Greedy autoregressive generation."""
        self.eval()
        for _ in range(max_new):
            x = prompt[:, -self.config.max_seq_len:]
            logits, _ = self.forward(x)
            next_logit = logits[:, -1, :]              # last token
            next_id = next_logit.argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_id], dim=1)
        return prompt


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING LOOP  (synthetic task: copy sequence)                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def make_synthetic_data(config: MoEConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple copy task: input = random bytes, target = same bytes.
    """
    vocab_size = config.vocab_size
    # Reserve tokens 0 (pad), 1 (bos), 2 (eos) — rest are data
    data_tokens = vocab_size - 3

    def _one_batch():
        x = torch.randint(3, vocab_size, (config.batch_size, config.max_seq_len))
        return x, x.clone()  # input = target (copy task)

    return _one_batch


def train(config: MoEConfig):
    """Run a short training loop and report metrics."""
    model = MoETransformer(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    data_fn = make_synthetic_data(config)

    print(f"{'Step':>5}  {'Loss':>8}  {'CE':>8}  {'Aux':>8}  {'Acc':>8}")
    print("-" * 45)

    for step in range(1, config.num_steps + 1):
        x, y = data_fn()
        x, y = x.to(config.device), y.to(config.device)

        logits, aux_loss = model(x)
        # Cross-entropy
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

        if step % 20 == 0 or step == 1:
            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(
                f"{step:5d}  {total_loss.item():8.4f}  "
                f"{ce_loss.item():8.4f}  {aux_loss.item():8.4f}  {acc:8.4f}"
            )

    return model


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER ANALYSIS  (diagnostics to evaluate load balance, etc.)            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def analyse_router(model: MoETransformer, num_batches: int = 5):
    """
    Run a few batches and report how evenly tokens are distributed across
    experts.  A well‑balanced router will have each expert handling ~
    (batch * seq * top_k / num_experts) tokens.

    Patent‑relevant: if your new router achieves better balance or more
    interesting specialisation, this function will prove it.
    """
    config = model.config
    model.eval()
    data_fn = make_synthetic_data(config)

    expert_counts = torch.zeros(config.num_experts)
    total_tokens = 0

    for _ in range(num_batches):
        x, _ = data_fn()
        x = x.to(config.device)
        _, aux_loss = model(x)

        # Extract routing decisions from the first layer
        layer = model.layers[0]
        moe = layer.moe
        with torch.no_grad():
            # The router expects hidden states [B, T, D], not token IDs.
            # Embed tokens first to get proper hidden states.
            hidden = model.token_embedding(x) + model.pos_embedding[:, :x.size(1), :]
            # Also apply the layer norm that feeds into the MoE
            hidden = layer.ln2(hidden)
            _, expert_indices, _ = moe.router(hidden)
            # expert_indices: [batch*seq, top_k]

        for i in range(config.num_experts):
            expert_counts[i] += (expert_indices == i).sum().item()
        total_tokens += x.numel()

    print("\n=== Router Analysis ===")
    print(f"Total token assignments: {int(total_tokens * config.top_k)}")
    print(f"Per-expert token count:  {expert_counts.tolist()}")
    print(f"Perfect balance would be: {total_tokens * config.top_k / config.num_experts:.1f} each")

    # Load imbalance metric (higher = worse)
    ideal = total_tokens * config.top_k / config.num_experts
    imbalance = (expert_counts - ideal).abs().sum().item() / ideal
    print(f"Load imbalance ratio: {imbalance:.4f}  (0 = perfect balance)")
    print()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    config = MoEConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {config.device}")
    print(f"Model params: ~? (counting...)")
    model = MoETransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Training ─────────────────────────────────────────────────────
    model = train(config)

    # ── Inference demo ───────────────────────────────────────────────
    model.eval()
    prompt = torch.randint(3, config.vocab_size, (1, 8))
    output = model.generate(prompt, max_new=16)
    print(f"\nPrompt:  {prompt[0].tolist()}")
    print(f"Generated: {output[0].tolist()}")

    # ── Router diagnostics ───────────────────────────────────────────
    analyse_router(model, num_batches=5)

    # ── Save ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "moe_model.pt")
    print("Model saved to moe_model.pt")
