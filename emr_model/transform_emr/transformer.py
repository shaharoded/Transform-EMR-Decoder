"""
transformer.py
==============

GPT wrapper that plugs into the project-wide Time2Vec + context
embedding defined in embedding.py and the batch structure produced
by dataset.py.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend
from pathlib import Path
from tqdm.auto import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.embedder import EMREmbedding
from transform_emr.config.model_config import *
from transform_emr.utils import *
from transform_emr.loss import MaskedFocalBCE, MaskedSetCE, pairwise_ranking_loss
from transform_emr.schedulers import LambdaScheduleController, LRScheduleController

# ───────── components  ───────────────────────────────────────────────── #
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with temporal RoPE."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg["embed_dim"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["embed_dim"]

        self.qkv   = nn.Linear(cfg["embed_dim"], 3 * cfg["embed_dim"], bias=cfg["bias"])
        self.proj  = nn.Linear(cfg["embed_dim"], cfg["embed_dim"],    bias=cfg["bias"])
        self.attn_dropout  = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])
        self.rope_t_scale = cfg.get("rope_t_scale", 24.0)

        # Task 4A: learned scalar temporal bias added to pre-softmax attention logits.
        # g(Δt_ij) = w * log1p(|Δt_ij| hours) + b  — initialized to zero (no-op at Phase-2 start).
        self.time_bias_w = nn.Parameter(torch.zeros(1))
        self.time_bias_b = nn.Parameter(torch.zeros(1))

    def _apply_temporal_rope(self, x, abs_ts):
        """Rotate Q/K by timestamp-dependent phases before dot-product attention.

        This injects absolute time into attention scores by phase-shifting each
        head channel pair with frequencies scaled by ``rope_t_scale``.
        """
        _, _, _, hd = x.shape
        half = hd // 2
        freq = 1.0 / (self.rope_t_scale ** (torch.arange(half, device=x.device, dtype=x.dtype) / half))
        theta = abs_ts.unsqueeze(-1).to(x.dtype) * freq  # [B, T, half]
        theta = theta.unsqueeze(1)  # [B, 1, T, half] broadcast over heads
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

    def forward(self, x, key_pad_mask=None, abs_ts=None, past_kv=None, use_cache=False):
            """
            Args:
                x            : [B, T, C]
                key_pad_mask : [B, T_kv] bool (True=keep).  For normal forward T_kv==T;
                               for KV-cache decode it spans the full cached+new key sequence.
                abs_ts       : [B, T] absolute timestamps for temporal RoPE.
                past_kv      : (k_cache, v_cache) each [B, n_head, T_past, hd].
                               When provided, x must be only the *new* tokens (T typically 1).
                               New K/V are appended and the full sequence is used for attention.
                use_cache    : If True, return (output, (k_full, v_full)) instead of just output.

            Returns:
                y            : [B, T, C]
                (k_full, v_full) : only when use_cache=True — updated KV cache tensors,
                                   each [B, n_head, T_past+T, hd].
            """
            B, T, C = x.shape
            qkv = self.qkv(x)                      # [B, T, 3C]
            q, k, v = qkv.split(C, dim=-1)         # each [B, T, C]

            # -- reshape to (B, h, T, d) so SDPA attends along time --
            hd = C // self.n_head
            q = q.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)
            k = k.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)
            v = v.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)

            # Apply temporal RoPE to Q and K.
            # NOTE: when past_kv is provided, abs_ts contains only the NEW token timestamps.
            # Past K values are already pre-rotated in the cache — valid because each K_i was
            # rotated by its own timestamp when first computed.
            if abs_ts is not None:
                q = self._apply_temporal_rope(q, abs_ts)
                k = self._apply_temporal_rope(k, abs_ts)

            # Task 4A: temporal bias matrix g(Δt_ij) = w*log1p(|Δt_ij|_hours) + b.
            # Only computed in prefill/training paths (past_kv is None) where full abs_ts [B,T] is available.
            # Compute log1p in BF16 with autocast disabled so the saved-for-backward tensor
            # stays BF16 (PyTorch autocast otherwise promotes log1p to FP32, doubling memory
            # for the [B, T, T] intermediate at every layer).
            time_bias = None
            if abs_ts is not None and past_kv is None:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    dt_hours = ((abs_ts.unsqueeze(-1) - abs_ts.unsqueeze(-2)) * 336.0).to(q.dtype).abs()  # [B, T, T]
                    dt_log   = torch.log1p(dt_hours)                                                       # BF16
                    time_bias = (self.time_bias_w.to(q.dtype) * dt_log + self.time_bias_b.to(q.dtype)).unsqueeze(1)  # [B, 1, T, T]

            # SDPA backend preference: mem-efficient first (avoids materialising the
            # full [B, h, T, T] attention-weight matrix for backward), math as fallback.
            # Flash is excluded because it does not accept attn_mask (we always have one
            # in training due to padding + temporal bias).
            _sdpa_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

            # ── KV-cache decode path ──────────────────────────────────────────
            if past_kv is not None:
                k_past, v_past = past_kv
                k = torch.cat([k_past, k], dim=2)   # [B, h, T_past+T, hd]
                v = torch.cat([v_past, v], dim=2)

                # key_pad_mask covers the full [T_past+T] key sequence.
                # Q [B, h, T, hd] attends to all past+new keys — no causal mask needed
                # since T is the newest position(s) and everything in cache is older.
                if key_pad_mask is not None:
                    T_kv      = k.shape[2]
                    pad_m     = (~key_pad_mask).unsqueeze(1).unsqueeze(2)    # [B,1,1,T_kv]
                    attn_mask = torch.zeros(B, 1, T, T_kv, device=x.device, dtype=q.dtype)
                    attn_mask.masked_fill_(pad_m, float("-inf"))
                    with sdpa_kernel(_sdpa_backends):
                        attn = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask, is_causal=False,
                            dropout_p=self.attn_dropout.p if self.training else 0.0,
                        )
                else:
                    with sdpa_kernel(_sdpa_backends):
                        attn = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=None, is_causal=False,
                            dropout_p=self.attn_dropout.p if self.training else 0.0,
                        )

            # ── Normal (prefill / training) path ─────────────────────────────
            elif key_pad_mask is not None:
                # --- Manual Masking Path (Required when padding is present) ---
                # SDPA throws RuntimeError if we pass both attn_mask and is_causal=True.
                # We must construct a combined mask [B, 1, T, T] containing both:
                # 1. Padding constraints (Cols)
                # 2. Causal constraints (Upper Triangle)

                # 1. Padding Mask: [B, 1, 1, T] -> Broadcast to [B, 1, T, T]
                # key_pad_mask is True(Keep), False(Pad). ~key_pad_mask is True(Mask Out).
                padding_mask = (~key_pad_mask).unsqueeze(1).unsqueeze(2)

                # 2. Causal Mask: [1, 1, T, T]
                # triu(1) gives upper triangle (future) as True -> Mask Out.
                causal_mask = torch.ones((T, T), device=x.device, dtype=torch.bool).triu(1).view(1, 1, T, T)

                # 3. Combine: Mask out if it's Padding OR Future
                combined_mask = padding_mask | causal_mask

                # 4. Convert to float mask for SDPA (-inf for mask, 0.0 for keep)
                # Using zeros_like ensures we match dtype (e.g. float16/bfloat16)
                attn_mask = torch.zeros_like(combined_mask, dtype=q.dtype).masked_fill(combined_mask, float("-inf"))

                if time_bias is not None:
                    attn_mask = attn_mask + time_bias

                with sdpa_kernel(_sdpa_backends):
                    attn = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=self.attn_dropout.p if self.training else 0.0,
                        is_causal=False  # We handled causality manually in the mask
                    )
            else:
                # --- Optimized Path (No padding, e.g. Inference) ---
                if time_bias is not None:
                    # Must build causal mask explicitly to add temporal bias.
                    causal_mask = torch.ones((T, T), device=x.device, dtype=torch.bool).triu(1).view(1, 1, T, T)
                    attn_mask = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
                    attn_mask.masked_fill_(causal_mask, float("-inf"))
                    attn_mask = attn_mask + time_bias
                    with sdpa_kernel(_sdpa_backends):
                        attn = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=self.attn_dropout.p if self.training else 0.0,
                            is_causal=False
                        )
                else:
                    # Here we can let SDPA handle the causal mask efficiently
                    attn = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=None,
                        dropout_p=self.attn_dropout.p if self.training else 0.0,
                        is_causal=True
                    )

            y = attn.transpose(1, 2).contiguous().view(B, T, C)         # -> [B, T, C]
            y = self.proj(y)
            y = self.resid_dropout(y)

            if use_cache:
                return y, (k, v)
            return y


class MLP(nn.Module):
    """
    SwiGLU MLP (SiLU Gating).
    Projects to a higher dimension, splits into Gate and Value, and multiplies them.
    Standard optimization for modern LLMs (LLaMA-style).
    """
    def __init__(self, cfg):
        super().__init__()
        # Standard GPT uses 4x expansion.
        # For SwiGLU, we project to 2 * (4 * dim) so we can split it into two 4x vectors.
        hidden_dim = 4 * cfg["embed_dim"]

        self.w1 = nn.Linear(cfg["embed_dim"], 2 * hidden_dim, bias=cfg["bias"])
        self.w2 = nn.Linear(hidden_dim, cfg["embed_dim"], bias=cfg["bias"])
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        # 1. Project to double width
        projected = self.w1(x)

        # 2. Split into x (content) and g (gate)
        x_val, x_gate = projected.chunk(2, dim=-1)

        # 3. Gating: Val * SiLU(Gate)
        out = x_val * F.silu(x_gate)

        # 4. Project back
        return self.drop(self.w2(out))


class AdaLNBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning.

    Instead of adding [CTX] to the sequence, we inject patient context into
    the normalization layers of *every* block.

    Mechanism:
        1. Project context_emb -> (scale, shift, gate)
        2. Norm(x) = (x - mu)/sigma * (1 + scale) + shift
        3. Output = x + gate * Block(Norm(x))

    Zero Initialization:
        We initialize the modulation projection to 0. This ensures that at the
        start of training, the block acts as an identity function (ignoring context),
        which stabilizes deep training.
    """
    def __init__(self, cfg):
            super().__init__()
            self.att = CausalSelfAttention(cfg)
            self.mlp = MLP(cfg)

            # 1. Norms WITHOUT affine parameters (we predict them from context)
            self.ln1 = nn.LayerNorm(cfg["embed_dim"], elementwise_affine=False, eps=1e-6)
            self.ln2 = nn.LayerNorm(cfg["embed_dim"], elementwise_affine=False, eps=1e-6)

            # 2. Modulation Head
            # Predicts 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cfg["embed_dim"], 6 * cfg["embed_dim"], bias=True)
            )

            # 3. Zero Init
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond_emb, key_pad_mask=None, abs_ts=None, past_kv=None, use_cache=False):
        """
        Calculate modulation parameters [B, 6*D] -> 6 x [B, D] (broadcast over T)
        x: [B, T, D]
        cond_emb: [B, D] (Patient Context)
        past_kv / use_cache: threaded through to CausalSelfAttention for KV caching.

        Returns:
            x             : [B, T, D]
            (k_full, v_full) : only when use_cache=True.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond_emb).chunk(6, dim=1)
        )

        # -- Attention Sub-block --
        norm_x   = self.ln1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.att(norm_x, key_pad_mask=key_pad_mask, abs_ts=abs_ts,
                            past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            attn_out, new_kv = attn_out

        x = x + gate_msa.unsqueeze(1) * attn_out

        # -- MLP Sub-block --
        norm_x = self.ln2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)

        if use_cache:
            return x, new_kv
        return x


# ───────── the GPT wrapper that consumes EMREmbedding ─────────────────────── #
class GPT(nn.Module):
    """
    GPT-style decoder that takes an *external* EMREmbedding instead of its own
    token/positional embeddings.

    The model learns the contextual connections between events in the EMR, and generates a
    predicted stream of events, from which the expected complications are derived.

    Parameters
    ----------
    cfg            : dict - hyper-parameters (n_layer, n_head, dropout, ...)
    embedder       : EMREmbedding - fully initialised shared embedding module
    use_checkpoint : bool - continue training from last checkpoint
    """

    def __init__(self, cfg: dict, embedder: EMREmbedding, use_checkpoint: bool=True):
        super().__init__()

        assert cfg["embed_dim"] == embedder.output_dim, (
            "Config embed_dim must equal EMREmbedding.output_dim"
        )

        self.cfg      = cfg
        self.embedder = embedder
        self.use_checkpoint = use_checkpoint

        # ─── Sanity checks ─────────────────────────────────────────────────────────────
        vocab_size = self.embedder.decoder.out_features

        assert hasattr(self.embedder.tokenizer, "id2token"), "[GPT] Embedder missing id2token map"
        assert len(self.embedder.tokenizer.id2token) == vocab_size, (
            f"[GPT] id2token size mismatch: got {len(self.embedder.tokenizer.id2token)}, expected {vocab_size}"
        )
        assert len(self.embedder.tokenizer.token2id) == self.embedder.position_embed.num_embeddings, \
            f"[GPT] Mismatch between tokenizer (len={len(self.embedder.tokenizer.token2id)}) and position_embed ({self.embedder.position_embed.num_embeddings})"

        # ─── Build layers ─────────────────────────────────────────────────────────────
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["embed_dim"], eps=1e-5) # Final Norm (Standard LayerNorm with learnable affine)

        # Next token prediction head (What will be the next event?)
        self.lm_head = nn.Linear(cfg["embed_dim"], vocab_size, bias=False)
        self.lm_head.weight = self.embedder.position_embed.weight  # weight tying
        assert self.lm_head.weight.shape[0] == vocab_size, (
            f"[GPT] lm_head output dim ({self.lm_head.weight.shape[0]}) "
            f"does not match embedder.position_embed ({vocab_size})"
        )

        # Outcome Head (designed to propogate signal to the hidden state that there is an expected outcome soon)
        # This head maps the embedding dimension to the unique outcome classes (e.g., Death, Sepsis, Hypoglycemia, Release)
        all_config_outcomes = sorted(list(set(OUTCOMES + TERMINAL_OUTCOMES)))

        # Filter by tokenizer vocabulary, then by tokenizer's pre-computed rarity filter.
        # The tokenizer decides which outcomes are valid (applied once at build time).
        tok = self.embedder.tokenizer
        in_vocab = [n for n in all_config_outcomes if n in tok.token2id]
        missing_outcomes = [n for n in all_config_outcomes if n not in tok.token2id]
        if missing_outcomes:
            print(f"[GPT] Outcomes not in tokenizer vocab (ignored): {missing_outcomes}")

        # outcome_patient_ratios keys are the valid outcomes; empty dict = old tokenizer, use all
        if getattr(tok, 'outcome_patient_ratios', None):
            valid_set = set(tok.outcome_patient_ratios.keys())
            valid_outcomes = [n for n in in_vocab if n in valid_set]
        else:
            valid_outcomes = in_vocab

        # Hard Error if nothing is left (Training cannot proceed)
        if not valid_outcomes:
            raise ValueError(
                f"[GPT] No valid outcomes found! Configured outcomes: {all_config_outcomes}. "
                "None of these exist in the tokenizer vocabulary. Check your dataset configuration or tokenizer build."
            )
        self.outcome_names = valid_outcomes
        self.num_outcomes = len(self.outcome_names)

        # 2-layer MLP outcome classifier. Adding more depth / width was tested
        # and consistently hurt by delaying the curriculum's stage-1 unlock.
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["embed_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["embed_dim"], self.num_outcomes)
        )

        # Vocab positions for ALL OUTCOMES+TERMINAL_OUTCOMES that exist in vocab.
        # Broader than outcome_names (includes outcomes filtered by rarity); used
        # for the 1-hot override in multi-hot BCE targets during phase-2 training.
        self.register_buffer(
            "_outcome_ids",
            torch.tensor([tok.token2id[n] for n in in_vocab], dtype=torch.long),
            persistent=True,
        )

        # Terminal token ids — get_temporal_multi_hot_targets applies a wider
        # future BCE window for these so the LM head sees many pre-terminal
        # positions as terminal-positive instead of only the immediate-next one.
        self.register_buffer(
            "_terminal_ids",
            torch.tensor(
                [tok.token2id[n] for n in TERMINAL_OUTCOMES if n in tok.token2id],
                dtype=torch.long,
            ),
            persistent=True,
        )

        # Per-outcome learnable log-tau for the outcome-head soft target. Each
        # outcome learns its own decay constant — RELEASE's healthy-patient
        # dynamics need a different tau than the clinical complications.
        _init_log_tau = math.log(12.0 / 336.0)
        self.outcome_log_tau = nn.Parameter(torch.full((self.num_outcomes,), _init_log_tau))

        # Direction C: per-token-class learnable log-tau for the LM-head
        # multi-hot BCE soft kernel. Replaces the hard two-tier window
        # (12h default / 168h terminals from exp59). Three-tier init aligned
        # with how each class is used downstream (exp73):
        #   - default tokens          → log(12 / 336)  (~12h)
        #   - outcome-class tokens    → log(48 / 336)  (matches outcome_horizon_hours)
        #   - terminal tokens         → log(168 / 336) (matches exp59 wide-window)
        # The outcome-class tier means CARDIO / KIDNEY / HYPER / HYPOGLY etc.
        # start with a wider kernel that aligns with the eval window. exp71
        # showed CARDIO regressed −0.077 when P2 outcome BCE was removed;
        # the hypothesis is that the default 12h init was too narrow for
        # complications. Model still learns per-class scale from these inits.
        _log_tau_default  = math.log(12.0  / 336.0)
        _log_tau_outcome  = math.log(48.0  / 336.0)
        _log_tau_terminal = math.log(168.0 / 336.0)
        _log_tau_lm = torch.full((vocab_size,), _log_tau_default)
        if self._outcome_ids.numel() > 0:
            _log_tau_lm[self._outcome_ids] = _log_tau_outcome
        if self._terminal_ids.numel() > 0:
            _log_tau_lm[self._terminal_ids] = _log_tau_terminal
        self.log_tau_lm = nn.Parameter(_log_tau_lm)

        # Δt prediction: two-head gate + magnitude design
        # Head 1 (gate): binary classifier P(Δt > 0) — handles 78.6% simultaneous events
        # Head 2 (magnitude): regression of Δt magnitude for non-zero cases only
        self.dt_gate = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["time2vec_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["time2vec_dim"], 1)
            # Output: logit for P(Δt > 0)
        )
        self.dt_magnitude = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["time2vec_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["time2vec_dim"], 1)
            # Output: raw magnitude (softplus applied later)
        )

        self.apply(self._init_weights)
        # Slightly smaller init for residual projections as in GPT-2:
        # scale down the output projection of each residual branch by 1/√(2*n_layers)
        # to keep the residual stream variance bounded at init in deep networks.
        # Targets: att.proj (attention output) and mlp.w2 (MLP output).
        for n, p in self.named_parameters():
            if n.endswith(("att.proj.weight", "mlp.w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

        print(f"[GPT]: Total params: {self.get_num_params()/1e6:.2f} M")


    # -------------------------------------------------------- helpers ------- #
    def _init_weights(self, module):
        """
        Custom initialization to ensure stable training.
        Method based on GPT2 initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """
        Utility method to get the number of parameters in the chosen architecture.
        """
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Configures the optimizer:
        - Applies weight decay to transformer weights (dim ≥ 2)
        - No weight decay on biases / norms
        - Applies same LR everywhere, but scales embedder LR by 0.1
        """
        embedder_params = list(self.embedder.parameters())
        embedder_param_ids = set(id(p) for p in embedder_params)

        decay, no_decay = [], []

        for n, p in self.named_parameters():
            if not p.requires_grad or id(p) in embedder_param_ids:
                continue  # Embedder handled separately
            (decay if p.dim() >= 2 else no_decay).append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": learning_rate},
            {"params": embedder_params, "weight_decay": 0.0, "lr": learning_rate * 0.1} # Lower LR for embedder tweaks
        ]

        return torch.optim.AdamW(optim_groups, betas=betas)

# ---------------------------------------------------- forward ---- #
    def forward(self, parent_raw_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec=None):
        """
        All tensors come straight from `collate_emr`:
            parent_raw_ids (torch.Tensor)    - padded raw_concept ids, (B, T, D)
            concept_ids (torch.Tensor)       - padded concepts ids, (B, T)
            value_ids (torch.Tensor)         - padded concept_value ids, (B, T)
            position_ids (torch.Tensor)      - padded token ids, (B, T)
            abs_ts (torch.Tensor)            - relative start times from ADMISSION (hours), (B, T)
            context_vec (torch.Tensor)       - age/gender or [] if not used, (B, C)

        Returns:
            logits (FloatTensor): [B, T, V]
                - Next-token logits for the sequence.
                - logits[b, t] predicts input position_ids[b, t+1].
                - The last logit logits[b, -1] predicts the future after the sequence ends.
            abs_t_pred (FloatTensor): [B, T]
                - Predicted absolute time for the *next* event.
                - abs_t_pred[b, t] predicts the time for the event at t+1.
            outcome_logits (FloatTensor): [B, T, self.num_outcomes]
                - Logits over the predefined outcomes in the task.
                - Logits are propagated to the hidden layers, informing the model on expected outcomes.
        """
        # 1. Embed (x=[B, T, D], cond=[B, D])
        # The embedder now returns the sequence and the separate context embedding for AdaLN blocks
        x, cond_emb, pad_mask = self.embedder(
            parent_raw_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec, return_mask=True
        )
        x = self.drop(x)

        # 2. Blocks with Conditioning (AdaLN)
        # We pass cond_emb to every block to modulate the normalization layers.
        # Gradient checkpointing is gated on torch.is_grad_enabled() rather than
        # self.training so Phase-3 fine-tune (backbone in eval() but params still
        # requires_grad=True with backbone_lr_factor>0) also gets the activation-memory
        # benefit. With dropout off (eval mode), recomputation is deterministic.
        _ckpt_active = torch.is_grad_enabled() and self.use_checkpoint
        for blk in self.blocks:
            if _ckpt_active:
                # use_reentrant=True: reentrant checkpoint, no tensor-count consistency check.
                # use_reentrant=False triggers CheckpointError on this model regardless of AMP
                # state (confirmed: amp=True for all 8 forward+recomp calls, counts still differ).
                # Default-arg capture (_blk=blk) fixes the closure bug where lambdas all closed
                # over the last block; reentrant recomputation runs correctly with each block's
                # own weights. Inner autocast ensures BF16 ops during recomputation.
                def _ckpt(_x, _c, _m, _t, _blk=blk):
                    _dev = _x.device.type
                    _amp = _dev == "cuda" and torch.cuda.is_bf16_supported()
                    with torch.autocast(device_type=_dev, dtype=torch.bfloat16, enabled=_amp):
                        return _blk(_x, _c, key_pad_mask=_m, abs_ts=_t)
                x = checkpoint.checkpoint(_ckpt, x, cond_emb, pad_mask, abs_ts, use_reentrant=True)
            else:
                x = blk(x, cond_emb, key_pad_mask=pad_mask, abs_ts=abs_ts)

        x = self.ln_f(x)                     # [B, T, D]

        # 3. Main next-token prediction head
        logits = self.lm_head(x)             # [B, T, V]

        # 4. Outcome Prediction Head (Auxiliary Task)
        outcome_logits = self.outcome_head(x)           # [B, T, K]

        # 5. Absolute time prediction (two-head: gate + magnitude)
        gate_logit = self.dt_gate(x).squeeze(-1)       # [B,T] logit for P(Δt>0)
        mag_raw = self.dt_magnitude(x).squeeze(-1)     # [B,T]
        mag_pos = F.softplus(mag_raw)                  # non-negative magnitude

        gate_prob = torch.sigmoid(gate_logit)
        delta_pos = gate_prob * mag_pos
        abs_t_pred = abs_ts + delta_pos                # [B, T]

        return logits, abs_t_pred, outcome_logits, gate_logit


    @torch.no_grad()
    def forward_with_cache(self, parent_raw_ids, concept_ids, value_ids, position_ids,
                           abs_ts, context_vec, past_kvs=None, cache_key_pad_mask=None):
        """
        KV-cache-aware forward pass for efficient batched autoregressive decoding.

        Two modes
        ---------
        **Prefill** (``past_kvs=None``) — process the full seed sequence, same as ``forward()``.
            All inputs are [B, T_seed, …].  Returns per-layer KV caches and the full logits
            so the caller can extract logits at the last *valid* position of each sequence.

        **Decode** (``past_kvs`` provided) — process a single new token per patient.
            All inputs are [B, 1, …].  The new token's K/V are appended to each layer's cache
            and only the new position's logits are returned.

        In both modes the return signature is identical:
            logits          [B, T_out, V]
            abs_t_pred      [B, T_out]        predicted time for the *next* event
            outcome_logits  [B, T_out, K]
            gate_logit      [B, T_out]
            new_kvs         List[Tuple[Tensor, Tensor]]  — one (k, v) per transformer layer
                            k/v shape [B, n_head, T_past+T_out, hd]

        where T_out == T_seed during prefill, and T_out == 1 during decode.

        Args:
            parent_raw_ids     : [B, T, P]
            concept_ids        : [B, T]
            value_ids          : [B, T]
            position_ids       : [B, T]
            abs_ts             : [B, T]
            context_vec        : [B, ctx_dim]
            past_kvs           : None | List[Tuple[k, v]]  (one tuple per layer)
            cache_key_pad_mask : [B, T_past+T] bool (True=valid).  Only used in decode mode.
                                 During prefill the pad mask is derived from position_ids.
        """
        is_decode = past_kvs is not None

        # 1. Embed
        x, cond_emb, pad_mask = self.embedder(
            parent_raw_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec, return_mask=True
        )
        x = self.drop(x)

        # 2. Blocks — collect new KV caches from every layer
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            pk  = past_kvs[i] if is_decode else None
            kpm = cache_key_pad_mask if is_decode else pad_mask
            x, new_kv = blk(x, cond_emb, key_pad_mask=kpm, abs_ts=abs_ts,
                            past_kv=pk, use_cache=True)
            new_kvs.append(new_kv)

        x = self.ln_f(x)

        # 3. Heads
        logits         = self.lm_head(x)
        outcome_logits = self.outcome_head(x)

        gate_logit = self.dt_gate(x).squeeze(-1)
        mag_pos    = F.softplus(self.dt_magnitude(x).squeeze(-1))
        delta_pos  = torch.sigmoid(gate_logit) * mag_pos
        abs_t_pred = abs_ts + delta_pos

        return logits, abs_t_pred, outcome_logits, gate_logit, new_kvs


    def save(self, path, epoch=None, best_val=None, optimizer=None, scheduler=None,
             lambda_schedule_state=None, training_settings=None, bad_epochs=0):
        ckpt = {
            "model_state": self.state_dict(),
            "config": copy.deepcopy(self.cfg),
            "vocab_size": self.embedder.decoder.out_features,
            "outcome_names": self.outcome_names,
            "num_outcomes": self.num_outcomes,
            "lambda_schedule_state": lambda_schedule_state,
            # Keep a copy of training settings used to produce this checkpoint.
            "training_settings": copy.deepcopy(training_settings),
            "bad_epochs": bad_epochs,
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if best_val is not None:
            ckpt["best_val"] = best_val
        if optimizer is not None:
            ckpt["optim_state"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state"] = scheduler.state_dict()
        torch.save(ckpt, path)


    @classmethod
    def load(cls, path, embedder, map_location=None):
        import torch as _t
        if map_location is None:
            map_location = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=map_location, weights_only=True)

        if "config" not in ckpt:
            raise ValueError("[GPT.load] Invalid checkpoint: missing 'config'.")

        # === Vocab safety check ===
        expected_vocab = ckpt["vocab_size"]
        actual_vocab = embedder.decoder.out_features
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[GPT.load] Embedder vocab size mismatch: checkpoint={expected_vocab}, embedder={actual_vocab}"
            )

        # === Outcome configuration check ===
        if "outcome_names" not in ckpt or "num_outcomes" not in ckpt:
            raise ValueError(
                "[GPT.load] Invalid checkpoint: missing 'outcome_names' or 'num_outcomes'. "
                "Checkpoint was saved with an older version of the code."
            )

        expected_outcome_names = set(ckpt["outcome_names"])
        expected_num_outcomes = ckpt["num_outcomes"]

        # Validate that current config matches checkpoint
        current_outcomes = set(OUTCOMES + TERMINAL_OUTCOMES)
        if not expected_outcome_names.issubset(current_outcomes):
            raise ValueError(
                f"[GPT.load] Outcome configuration mismatch!\n"
                f"  Checkpoint outcomes: {sorted(expected_outcome_names)}\n"
                f"  Current config outcomes: {sorted(current_outcomes)}\n"
                f"  Please use the same OUTCOMES and TERMINAL_OUTCOMES config that was used during training."
            )

        # Reconstruct model
        model = cls(cfg=ckpt["config"], embedder=embedder)

        # Final sanity check
        if model.num_outcomes != expected_num_outcomes:
            raise ValueError(
                f"[GPT.load] Architecture mismatch: checkpoint has {expected_num_outcomes} outcomes, "
                f"but reconstructed model has {model.num_outcomes}."
            )

        state_dict = ckpt["model_state"]
        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(state_dict.keys())
        missing    = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        if missing:
            raise RuntimeError(f"[GPT.load] Missing required keys in checkpoint: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"[GPT.load] Unexpected keys in checkpoint: {sorted(unexpected)}")
        model.load_state_dict(state_dict, strict=False)

        # Align device: the caller's embedder may already be on GPU while the checkpoint
        # was saved on CPU. Move the whole model to match the embedder's device so all
        # parameters and inputs are on the same device during forward passes.
        embedder_device = next(embedder.parameters()).device
        model.to(embedder_device)

        # Helpful metadata for callers that want to fully restore prior training settings.
        model.checkpoint_model_config = copy.deepcopy(ckpt["config"])
        model.checkpoint_training_settings = copy.deepcopy(ckpt.get("training_settings"))

        # Return full training state if available
        return (
            model,
            ckpt.get("epoch", 0),
            ckpt.get("best_val", float("inf")),
            ckpt.get("optim_state"),
            ckpt.get("scheduler_state"),
            ckpt.get("lambda_schedule_state"),
        )


@logger
def pretrain_transformer(model, train_dl, val_dl, resume=True, checkpoint_path=PHASE2_CHECKPOINT, training_settings=TRAINING_SETTINGS):
    """
    Trains a Transformer-based EMR sequence model in Phase 2 (decoder stage),
    using a pretrained embedder and structured multi-loss optimization.
    Total Loss = λ1 * BCE + λ2 * CE + λ3 * Outcome prediction + λt * Time Loss (τt)

    The auxiliary losses are applied gradually to stabilize training:
    -  Next-token BCE loss encourages accurate event prediction (foundational task, start at epoch 0 with no schedule).
    -  Next-token CE loss provides a complementary signal for token prediction (foundational task, start at epoch 0 with no schedule).
    -  Time prediction MSE loss guides the model to predict event timings (foundational task, start at epoch 0 with no schedule).
    -  Outcome prediction BCE loss guides the model to predict clinical outcomes (post-foundational task, start after the foundational phase with schedule).

    Gradually applying curriculum and CBM during `scheduler.bce_only_epochs`, then implements early stopping based on best total loss, after warmup concludes.
    Supports resume-from-checkpoint training. The CBM is applied during the foundational training phase only.

    Logits are masked to ensure only legal tokens are predicted, targets are masked to ensure proper loss denominator,
    and penalties are applied to encourage valid event sequences.

    Args:
        model (nn.Module): GPT decoder with attached EMREmbedding.
        train_dl (DataLoader): Training data loader.
        val_dl (DataLoader): Validation data loader.
        resume (bool): Resume from latest checkpoint if found.
        checkpoint_path (str): Path to save the best model and state.
        training_settings (dict): A settings dictionary, imported from model_config.

    Returns:
        tuple: (model, train_losses, val_losses)
        Saves model checkpoints and plots training curves.

    NOTE: Despite intention, this function currently not implementing curriculum. in order to do so, one must add after
    pred_ids are calculated (in run_epoch) the following additions:
    if train_flag -> mix_with_predictions() from utils.py -> rebuild raw/concept/value from predicted position (define LUT?) ->
    apply mix_mask to all modalities -> Second forward pass on mixed inputs.
    """
    # Create global training lookup Tensors once the tokenizer is available and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    luts = build_luts(model.embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k,v in luts.items()}

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    # If resuming, prefer checkpoint-saved settings to avoid config mismatch.
    if resume and ckpt_last.exists():
        pre_ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=True)
        if pre_ckpt.get("training_settings") is not None:
            training_settings = pre_ckpt["training_settings"]

    # Allow embedder weights to update starting epoch 1
    set_embedder_frozen(model, freeze=False)
    model.to(device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    optimizer = model.configure_optimizers(
        weight_decay=training_settings["weight_decay"],
        learning_rate=training_settings["phase2_learning_rate"],
        betas=(0.9, 0.95)
    )

    scheduler = LRScheduleController(optimizer, training_settings, train_dl)

    # Training Dynamics
    cbm_ramp_epochs = training_settings["phase2_scheduler"]["bce_only_epochs"] # Couple CBM ramp up with the foundational phase to stabilize training before introducing the full curriculum. CBM will ramp up from 0 to max_p during these epochs, then stay at max_p for the rest of training.

    valid_outcomes = [n for n in model.outcome_names if n in model.embedder.tokenizer.token2id]
    outcome_token_ids = [model.embedder.tokenizer.token2id[n] for n in valid_outcomes]

    # Build criterions once (for next token BCE and CE losses + aux outcome BCE)
    BCEcriterion = MaskedFocalBCE.from_counts(
        counts=model.embedder.tokenizer.token_counts,
        token_weights=model.embedder.tokenizer.token_weights,
        beta=0.999, min_count=5, clip_max=8.0,
        gamma=0.5,          # mild focal suppression — pretrained embedder already makes examples "easy"
        tau=0.85,           # pos/neg balance anchor
        neg_bounds=(0.03, 0.3),   # clamp for stability
        label_smoothing=0.0,     # optional
        hard_neg_k=0            # or e.g., 64 for hard-neg mining
    ).to(device)

    CEcriterion = MaskedSetCE(
        label_smoothing=0.0,     # optional
    ).to(device)

    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0

    lambda_schedule_state = None
    if resume and ckpt_last.exists():
        print(f"[GPT]: Loading model from checkpoint: {ckpt_last}")
        loaded_model, start_epoch, best_val, opt_state, sch_state, lambda_schedule_state = GPT.load(ckpt_last, embedder=model.embedder, map_location=device)
        model = loaded_model
        model.to(device)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

        # Prefer checkpoint-saved settings to avoid resume/config mismatch issues.
        if getattr(model, "checkpoint_training_settings", None) is not None:
            training_settings = model.checkpoint_training_settings

        # Scheduler needs to be re-initiated with the recovered optimizer before loading its state dict
        scheduler = LRScheduleController(optimizer, training_settings, train_dl)
        if sch_state is not None:
            scheduler.load_state_dict(sch_state)
        start_epoch += 1
        print(f"[Phase-2]: Resumed training at epoch {start_epoch} (best val_loss so far: {best_val:.4f})")
    else:
        print("[Phase-2]: Starting transformer training loop...")

    schedule_controller = LambdaScheduleController(
        schedule_config=training_settings["phase2_scheduler"],
        start_epoch=start_epoch
    )
    if lambda_schedule_state is not None:
        schedule_controller.load_state_dict(lambda_schedule_state)

    train_losses, val_losses = [], []

    grad_accum_steps = training_settings.get("grad_accumulation_steps", 1)

    def run_epoch(loader, epoch, train_flag=False):
        if train_flag:
            model.train()
        else:
            model.eval()

        total_loss = total_bce = total_ce = total_dt = total_ranking = 0.0
        total_ce_raw = total_dt_raw = total_ranking_raw = 0.0
        accum_step = 0
        if train_flag:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False, mininterval=5.0, miniters=10, dynamic_ncols=True):
                batch = {k: v.to(device) for k, v in batch.items()}

                # === Apply CBM on training batchs ===
                if train_flag:
                    # Starting at epoch 0, ramping through the foundational_epochs
                    p = linear_schedule(epoch=epoch,
                                        start_epoch=0,
                                        end_epoch=cbm_ramp_epochs,
                                        max_val=0.25)

                    batch = apply_cbm(batch=batch,
                                      tokenizer=model.embedder.tokenizer,
                                      forbid_ids=luts["forbid_mask_ids"],
                                      p=p)

                # === Original logits from Model ===
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits, abs_t_pred, outcome_logits, dt_gate_logit = model(
                        parent_raw_ids=batch["parent_raw_ids"],
                        concept_ids=batch["concept_ids"],
                        value_ids=batch["value_ids"],
                        position_ids=batch["position_ids"],
                        abs_ts=batch["abs_ts"],
                        context_vec=batch["context_vec"]
                    )
                logits = logits.float()
                abs_t_pred = abs_t_pred.float()
                outcome_logits = outcome_logits.float().clamp(-20.0, 20.0)
                dt_gate_logit = dt_gate_logit.float()

                # logits is [B, T, V]
                # abs_t_pred: [B, T]
                # outcome_logits: [B, T, K]
                # dt_gate_logit: [B, T] — logit for P(Δt > 0)

                # Slice for Autoregressive Training (Predict Next Token)
                # Input at t predicts Target at t+1.
                # We drop the LAST input (nothing to predict after it) and the FIRST target (nothing predicts it)

                # Slicing Logits (logits 0 to T-2)
                pred_logits = logits[:, :-1, :]       # [B, T-1, V]

                # Slicing Targets (Targets 1 to T-1)
                full_targets = batch["targets"]
                target_ids   = full_targets[:, 1:]    # [B, T-1] Used for loss calculation

                # === legality masks from Ground Truth + Targets ===
                # Compute on FULL targets to preserve state history (t=0)
                full_illegal = compute_legality_masks_tf(
                                      full_targets,
                                      luts["is_start"],
                                      luts["is_end"],
                                      luts["base_id"],
                                      luts["start_ids_per_base"],
                                      luts["end_ids_per_base"],
                                      luts["meal_rank"],
                                      luts["meal_pred_rank"],
                                      luts["K_meals"],
                                      luts["conflict_mat"],
                                      luts["predict_block"]
                                  )

                # Slice to align with target_ids (predict t+1 from position t)
                illegal_mask = full_illegal[:, 1:, :]

                nonpad = (target_ids != model.embedder.padding_idx)      # [B, T-1]

                # Suppress illegal tokens so BCE/CE never learns from them
                pred_logits = apply_masks_to_logits(pred_logits, illegal_mask)

                # Outcome Head Slicing
                # Input at t predicts future relative to t.
                # We align with the same truncated input sequence (0 to T-2).
                outcome_pred = outcome_logits[:, :-1, :] # [B, T-1, K]

                # Time Slicing
                # abs_t_pred[t] is the predicted time for token[t+1]
                # So abs_t_pred[0] should match abs_ts[1]
                pred_abs = abs_t_pred[:, :-1]     # [B, T-1]
                true_abs = batch["abs_ts"][:, 1:] # [B, T-1] (Shifted true times)

                # === Loss: BCE with logits — temporal targets ===
                # Per-token-class soft kernel: for each position t and class k, the
                # target weight is exp(-|Δt| / tau_k) over all future positions where
                # token == k, clamped to [0,1]. tau_k is learnable (model.log_tau_lm)
                # and horizon hard-zeros beyond phase2_terminal_bce_window_hours.
                _ABS_TS_SCALE = 336.0
                _TERM_WIN = training_settings.get("phase2_terminal_bce_window_hours", 168.0) / _ABS_TS_SCALE
                multi_hot = get_temporal_soft_targets(
                    target_ids=full_targets,
                    all_abs_ts=batch["abs_ts"],
                    query_abs_ts=batch["abs_ts"][:, :-1],
                    padding_idx=model.embedder.padding_idx,
                    vocab_size=pred_logits.size(-1),
                    tau=model.log_tau_lm.exp(),
                    horizon=_TERM_WIN,
                )
                multi_hot = multi_hot.masked_fill(illegal_mask, 0.0)

                # M: anti-terminal-dominance mask.
                # Zero out TERMINAL soft-labels at positions where the ground-truth
                # terminal token is >threshold hours away. This prevents Phase-2 BCE
                # from driving the backbone into the TERMINAL-dominant local minimum.
                # Without this, the BCE kernel (tau_terminal=168h) gives every position
                # high TERMINAL weight (most sequences end within 168h), teaching
                # TERMINAL at ALL positions — causing gen_median_steps=4 collapse.
                _term_thresh = training_settings.get("phase2_terminal_threshold_hours", None)
                if _term_thresh is not None and model._terminal_ids.numel() > 0:
                    _T_THRESH = _term_thresh / _ABS_TS_SCALE
                    # Find earliest terminal token time in each sequence
                    _is_term = torch.isin(full_targets, model._terminal_ids)  # [B, T]
                    _term_abs = batch["abs_ts"].clone()
                    _term_abs[~_is_term] = float("inf")
                    _first_term_t = _term_abs.min(dim=-1, keepdim=True).values  # [B, 1]
                    # Query times (the positions from which we're predicting)
                    _q_abs = batch["abs_ts"][:, :-1]                            # [B, T-1]
                    _dt_to_term = (_first_term_t - _q_abs).clamp(min=0)         # [B, T-1]
                    _not_imminent = _dt_to_term > _T_THRESH                     # [B, T-1]
                    for _tid in model._terminal_ids.tolist():
                        if 0 <= _tid < multi_hot.size(-1):
                            multi_hot[:, :, _tid] *= (~_not_imminent).float()

                # mask out illegal classes AND PAD steps from the denominator
                valid_pos = nonpad.unsqueeze(-1)            # [B,T,1] bool
                allowed   = (~illegal_mask) & valid_pos     # [B,T,V] bool

                # Calculate BCE loss (only valid positions)
                loss_bce, _ = BCEcriterion(pred_logits, multi_hot, allowed)

                # === Loss: CE nudge (next token generation task) ===
                # Foundational task to complement BCE, controlled by schedule
                loss_ce_raw, _ = CEcriterion(pred_logits, multi_hot, allowed)
                lambdas = schedule_controller.get_lambdas(epoch)
                loss_ce = loss_ce_raw * lambdas["ce"]

                # === Loss: Δt (time) — two-head: gate + magnitude ===
                # Gate: binary classification P(Δt > 0)
                true_dt = true_abs - batch["abs_ts"][:, :-1]  # [B, T-1]
                gate_pred = dt_gate_logit[:, :-1]              # [B, T-1]
                dt_nonzero = (true_dt > 1e-8)                  # bool mask for non-simultaneous
                gate_target = dt_nonzero.float()

                gate_loss = F.binary_cross_entropy_with_logits(
                    gate_pred[nonpad], gate_target[nonpad], reduction="mean"
                )

                # Magnitude: MSE on non-zero Δt only
                nonzero_mask = nonpad & dt_nonzero
                if nonzero_mask.any():
                    mag_loss = F.mse_loss(pred_abs[nonzero_mask], true_abs[nonzero_mask], reduction="mean")
                else:
                    mag_loss = torch.tensor(0.0, device=pred_abs.device)

                abs_t_loss_raw = gate_loss + mag_loss
                abs_t_loss = abs_t_loss_raw * lambdas["dt"]

                # === Outcome Loss — Time-Decayed Soft Labels ===
                # For each position t, the target for outcome k is a soft risk score:
                # sum_s { exp(-dt(t,s) / tau) * 1[token_s == outcome_k] }.clamp(0, 1)
                # Maximum gradient right before an outcome; decays to zero for distant/absent
                # outcomes. No hard window boundaries — decay handles separation naturally.
                _ABS_TS_SCALE = 336.0
                _HORIZON = training_settings.get("outcome_horizon_hours",   48.0) / _ABS_TS_SCALE
                # Per-outcome learnable tau (initialised at log(12h / _ABS_TS_SCALE)).
                _TAU = model.outcome_log_tau.exp()
                outcome_targets = get_future_outcome_targets(
                    target_ids=full_targets,
                    outcome_ids=outcome_token_ids,
                    all_abs_ts=batch["abs_ts"],
                    query_abs_ts=batch["abs_ts"][:, :-1],
                    tau=_TAU,
                    horizon=_HORIZON,
                )  # [B, T-1, K]

                # === Loss: Pairwise ranking (direct AUROC proxy on the outcome head) ===
                # Positive positions: outcome_targets > 0 (outcome occurs within horizon).
                # Negative positions: outcome_targets == 0 (no outcome within horizon).
                # Both masked by non-pad. Independent of soft-BCE — direct AUROC signal.
                # The raw loss is always computed so the lambda scheduler can calibrate it
                # at stage-1 unlock; gating on lam > 0 would deadlock calibration.
                _rank_pos = (outcome_targets > 0.0) & valid_pos
                _rank_neg = (outcome_targets == 0.0) & valid_pos
                loss_ranking_raw = pairwise_ranking_loss(
                    outcome_pred, _rank_pos, _rank_neg,
                )
                loss_ranking = lambdas.get("ranking", 0.0) * loss_ranking_raw

                # === Loss: Total Loss ===
                loss = loss_bce + loss_ce + abs_t_loss + loss_ranking

                # === Backprop and Log ===
                if train_flag:
                    # NaN guard: skip batch and log which component is bad.
                    # All-NaN collapse is typically caused by BF16 gradient overflow,
                    # not by gradual explosion (which clipping would catch).
                    _losses = {"bce": loss_bce, "ce": loss_ce, "dt": abs_t_loss, "ranking": loss_ranking, "total": loss}
                    _bad = {k: v.item() for k, v in _losses.items() if not torch.isfinite(v)}
                    if _bad:
                        print(f"[WARNING] Skipping batch (epoch {epoch}): non-finite losses={_bad}; zeroing grads.")
                        optimizer.zero_grad()
                        accum_step = 0
                        continue

                    # Backward in FP32 — no outer autocast needed with use_reentrant=True.
                    # Inner autocast inside _ckpt ensures BF16 recomputation matches forward.
                    (loss / grad_accum_steps).backward()

                    # Immediate per-batch NaN check: prevents NaN from one batch contaminating
                    # subsequent accumulation steps (NaN + finite = NaN accumulates silently).
                    _batch_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    if not torch.isfinite(_batch_norm):
                        optimizer.zero_grad()
                        accum_step = 0
                        continue

                    accum_step += 1
                    if accum_step % grad_accum_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.update()
                        optimizer.zero_grad()
                # Update loss
                total_loss    += loss.item()
                total_bce     += loss_bce.item()
                total_ce      += loss_ce.item()
                total_dt      += abs_t_loss.item()
                total_ranking += loss_ranking.item()
                total_ce_raw      += loss_ce_raw.item()
                total_dt_raw      += abs_t_loss_raw.item()
                total_ranking_raw += loss_ranking_raw.item()

        # Flush any remaining accumulated gradients at end of epoch
        if train_flag and accum_step % grad_accum_steps != 0:
            flush_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(flush_norm):
                optimizer.step()
                scheduler.update()
            else:
                print(f"[WARNING] NaN/inf grad norm at epoch-end flush (epoch {epoch}), skipping.")
            optimizer.zero_grad()

        n_batches = len(loader)

        return (
            total_loss    / n_batches,
            total_bce     / n_batches,
            total_ce      / n_batches,
            total_dt      / n_batches,
            total_ranking / n_batches,
            total_ce_raw      / n_batches,
            total_dt_raw      / n_batches,
            total_ranking_raw / n_batches,
        )

    for epoch in range(start_epoch, training_settings.get("phase2_n_epochs")):
        tr_loss, tr_bce, tr_ce, tr_dt, tr_ranking, tr_ce_raw, tr_dt_raw, tr_ranking_raw = run_epoch(train_dl, epoch=epoch, train_flag=True)
        vl_loss, vl_bce, vl_ce, vl_dt, vl_ranking, _, _, _                              = run_epoch(val_dl,   epoch=epoch, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"""[Phase-2]: Epoch {epoch:02d}
        --> Train={tr_loss:.4f} (BCE={tr_bce:.4f}, CE={tr_ce:.4f}, Δt={tr_dt:.4f}, Rank={tr_ranking:.4f})
        --> Val={vl_loss:.4f} (BCE={vl_bce:.4f}, CE={vl_ce:.4f}, Δt={vl_dt:.4f}, Rank={vl_ranking:.4f})
        --> RawTrain ce={tr_ce_raw:.7f} dt={tr_dt_raw:.7f} ranking={tr_ranking_raw:.7f}""")

        schedule_events = schedule_controller.update(
            epoch=epoch,
            vl_total=vl_loss,
            tr_main=tr_bce,
            ce=tr_ce_raw,
            dt=tr_dt_raw,
            ranking=tr_ranking_raw,
        )
        for msg in schedule_events:
            print(msg)
        if schedule_controller.has_dynamic:
            print(schedule_controller.status_line(epoch))

        # Save latest
        model.save(ckpt_last, epoch, best_val, optimizer, scheduler,
                   lambda_schedule_state=schedule_controller.state_dict(),
                   training_settings=training_settings, bad_epochs=bad_epochs)

        # Save best model
        warmup_gate = schedule_controller.current_warmup_end_epoch()

        min_delta_rel = training_settings.get("early-stop-min-delta-rel", 1e-3)
        if (vl_loss < best_val * (1.0 - min_delta_rel)) and (epoch >= warmup_gate):
            best_val = vl_loss
            model.save(ckpt_path, epoch, best_val, optimizer, scheduler,
                       lambda_schedule_state=schedule_controller.state_dict(),
                       training_settings=training_settings, bad_epochs=bad_epochs)
            print("[Phase-2]: Current best model saved.")
            bad_epochs = 0
        elif epoch >= warmup_gate:
            bad_epochs += 1
            if bad_epochs >= training_settings["early-stop-patience"]:
                print("[Phase-2]: Early stopping triggered.")
                break
        else:
            # If warmup isn't complete - do nothing.
            continue

    plot_losses(train_losses, val_losses)
    return model, train_losses, val_losses


@logger
def finetune_transformer(model, train_dl, val_dl, resume=True,
                         checkpoint_path=PHASE3_CHECKPOINT, training_settings=TRAINING_SETTINGS):
    """
    Phase 3 — Outcome Head Fine-tuning.

    Fine-tunes the outcome_head with differential learning rates:
      - outcome_head:  phase3_learning_rate      (e.g. 1e-4)
      - backbone:      phase3_learning_rate * phase3_backbone_lr_factor  (e.g. 1e-6)

    The backbone is kept in eval mode (deterministic features, no dropout) but receives
    tiny gradient updates guided by outcome loss. This allows micro-adaptation of the
    backbone without catastrophic forgetting. Setting phase3_backbone_lr_factor=0.0
    is equivalent to a full freeze.

    The outcome loss uses the same time-decayed soft labels as Phase 2 (same
    get_future_outcome_targets formula), so the head learns from exactly the same target
    distribution as Phase 2's outcome auxiliary.

    Because the backbone is frozen, this phase converges quickly. Early stopping is
    applied against validation outcome loss. Checkpoints are saved in the same full-model
    format as Phase 2, so GPT.load() works identically for both checkpoints.

    Args:
        model             : trained GPT (loaded from Phase-2 best checkpoint).
        train_dl          : training DataLoader (same batched loader used in Phase 2).
        val_dl            : validation DataLoader.
        resume            : if True, look for a Phase-3 checkpoint and continue from it.
        checkpoint_path   : path for the best-checkpoint file.
        training_settings : settings dict (from model_config).

    Returns
    -------
    model, train_losses, val_losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    tok = model.embedder.tokenizer
    valid_outcomes    = [n for n in model.outcome_names if n in tok.token2id]
    outcome_token_ids = [tok.token2id[n] for n in valid_outcomes]
    pos_weights       = tok.outcome_weights[outcome_token_ids].to(device)
    OutcomeCriterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")

    backbone_lr_factor = training_settings.get("phase3_backbone_lr_factor", 0.0)
    p3_lr = training_settings["phase3_learning_rate"]
    p3_wd = training_settings.get("phase3_weight_decay", training_settings["weight_decay"])

    def _freeze_backbone_only(m):
        """
        Enable gradients on all params; the optimizer's per-group LRs control the
        effective freeze (backbone LR is multiplied by phase3_backbone_lr_factor).

        outcome_log_tau is hard-frozen in Phase 3 — letting it drift here destroys
        the RELEASE signal Phase 2 built. The soft-label decay constants are a
        Phase-2 decision; Phase 3 only refines the classifier weights.
        """
        for param in m.parameters():
            param.requires_grad_(True)
        m.outcome_log_tau.requires_grad_(False)

    def _make_p3_optimizer(m):
        head_names = {"outcome_head"}
        backbone_params = [p for n, p in m.named_parameters()
                           if not any(h in n for h in head_names)]
        head_params = list(m.outcome_head.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": p3_lr * backbone_lr_factor,
                 "weight_decay": training_settings["weight_decay"]},
                {"params": head_params,     "lr": p3_lr,
                 "weight_decay": p3_wd},
            ],
        )

    _freeze_backbone_only(model)
    optimizer = _make_p3_optimizer(model)

    start_epoch = 1
    best_val    = float("inf")
    bad_epochs  = 0

    if resume and ckpt_last.exists():
        print(f"[Phase-3]: Loading checkpoint: {ckpt_last}")
        loaded_model, start_epoch, best_val, opt_state, _, _ = GPT.load(
            ckpt_last, embedder=model.embedder, map_location=device
        )
        model = loaded_model
        model.to(device)
        _freeze_backbone_only(model)
        optimizer = _make_p3_optimizer(model)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_epoch += 1
        print(f"[Phase-3]: Resumed at epoch {start_epoch} (best val: {best_val:.4f})")
    else:
        model.to(device)
        print(f"[Phase-3]: Fine-tuning outcome head (backbone_lr_factor={backbone_lr_factor})...")

    train_losses, val_losses = [], []

    _ABS_TS_SCALE = 336.0
    _HORIZON = training_settings.get("outcome_horizon_hours",   48.0) / _ABS_TS_SCALE
    # Read the learnable per-outcome tau from the model. In Phase 3 the param
    # continues to be trained alongside the outcome head.

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Mirror P2's pairwise ranking loss in Phase 3 so the outcome head stays in
    # the joint (outcome BCE + ranking) optimum it converged into at the end of
    # P2, rather than being pulled toward a BCE-only fit on the natural-
    # distribution loader. λ is calibrated once at the end of epoch 1 from the
    # epoch-1 raw outcome / raw ranking ratio, using the same fraction cap that
    # P2 uses for ranking — no new hyperparameter.
    #
    # Early-stop watches val_outcome_raw (not val_total): the joint training
    # loss surface changes at epoch-2 calibration (λ goes 0→λ_cal), so
    # val_total isn't comparable across that boundary. val_outcome_raw is
    # stable, and it is the metric the eventual eval cares about (outcome head
    # calibration on the natural distribution).
    ranking_cap = (training_settings.get("phase2_scheduler", {})
                                     .get("aux_fraction_caps", {})
                                     .get("ranking", 0.20))
    lambda_ranking: float | None = None  # set after epoch 1

    def run_epoch(loader, train_flag):
        # Backbone stays in eval mode (no dropout updates, deterministic features).
        model.eval()
        if train_flag:
            model.outcome_head.train()

        total_loss = total_outcome = 0.0
        total_outcome_raw = total_ranking_raw = 0.0
        with torch.set_grad_enabled(train_flag):
            for batch in tqdm(loader, desc="[Phase-3] Train" if train_flag else "[Phase-3] Val",
                              leave=False, mininterval=5.0, miniters=10, dynamic_ncols=True):
                batch = {k: v.to(device) for k, v in batch.items()}

                # BF16 autocast wraps the forward (and gradient-checkpointed recompute
                # inside GPT.forward also re-enters autocast for backward). Matches
                # Phase-2 precision regime and cuts activation memory roughly in half.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits, _, outcome_logits, _ = model(
                        parent_raw_ids=batch["parent_raw_ids"],
                        concept_ids=batch["concept_ids"],
                        value_ids=batch["value_ids"],
                        position_ids=batch["position_ids"],
                        abs_ts=batch["abs_ts"],
                        context_vec=batch["context_vec"],
                    )
                logits         = logits.float()
                outcome_logits = outcome_logits.float().clamp(-20.0, 20.0)

                full_targets = batch["position_ids"]      # [B, T]
                target_ids   = full_targets[:, 1:]        # [B, T-1]
                outcome_pred = outcome_logits[:, :-1, :]  # [B, T-1, K]
                pred_logits  = logits[:, :-1, :]          # [B, T-1, V]

                outcome_targets = get_future_outcome_targets(
                    target_ids=full_targets,
                    outcome_ids=outcome_token_ids,
                    all_abs_ts=batch["abs_ts"],
                    query_abs_ts=batch["abs_ts"][:, :-1],
                    tau=model.outcome_log_tau.exp(),
                    horizon=_HORIZON,
                )  # [B, T-1, K]

                nonpad    = (target_ids != model.embedder.padding_idx)  # [B, T-1]
                valid_pos = nonpad.unsqueeze(-1)                        # [B, T-1, 1]

                loss_outcome_raw = (OutcomeCriterion(outcome_pred, outcome_targets)
                                    * valid_pos).sum() / valid_pos.sum().clamp(min=1.0)
                loss_outcome = loss_outcome_raw

                # Pairwise ranking — same pos/neg construction as P2.
                _rank_pos = (outcome_targets > 0.0) & valid_pos
                _rank_neg = (outcome_targets == 0.0) & valid_pos
                loss_ranking_raw = pairwise_ranking_loss(outcome_pred, _rank_pos, _rank_neg)
                _lam = lambda_ranking if lambda_ranking is not None else 0.0
                loss_ranking = _lam * loss_ranking_raw

                loss = loss_outcome + loss_ranking

                if train_flag:
                    if not torch.isfinite(loss):
                        continue
                    optimizer.zero_grad()
                    loss.backward()
                    p3_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isfinite(p3_norm):
                        optimizer.step()

                total_loss        += loss.item()
                total_outcome     += loss_outcome.item()
                total_outcome_raw += loss_outcome_raw.item()
                total_ranking_raw += loss_ranking_raw.item()

        n = max(len(loader), 1)
        return {
            "loss":        total_loss        / n,
            "outcome":     total_outcome     / n,
            "outcome_raw": total_outcome_raw / n,
            "ranking_raw": total_ranking_raw / n,
        }

    n_epochs = training_settings["phase3_n_epochs"]
    patience = training_settings["early-stop-patience"]

    for epoch in range(start_epoch, start_epoch + n_epochs):
        tr = run_epoch(train_dl, train_flag=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vl = run_epoch(val_dl,   train_flag=False)

        # One-shot λ_ranking calibration at the end of epoch 1 (mirrors P2).
        if lambda_ranking is None and tr["ranking_raw"] > 0.0:
            lambda_ranking = ranking_cap * (tr["outcome_raw"] / tr["ranking_raw"])
            print(f"[Phase-3]: λ_ranking calibrated = {lambda_ranking:.6f} "
                  f"(cap={ranking_cap}, raw_outcome={tr['outcome_raw']:.6f}, "
                  f"raw_ranking={tr['ranking_raw']:.6f})")

        tr_loss, vl_loss = tr["loss"], vl["loss"]
        # Selection metric: pure outcome BCE (stable across the λ=0 → λ=λ_cal
        # transition). This is what the eval ultimately measures.
        vl_select = vl["outcome_raw"]
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Phase-3]: Epoch {epoch:02d}  train={tr_loss:.4f}  val={vl_loss:.4f}  "
              f"raw_out={tr['outcome_raw']:.7f}  raw_rank={tr['ranking_raw']:.7f}  "
              f"vl_select={vl_select:.6f}")

        model.save(ckpt_last, epoch=epoch, best_val=best_val, optimizer=optimizer,
                   training_settings=training_settings, bad_epochs=bad_epochs)

        min_delta_rel = training_settings.get("early-stop-min-delta-rel", 1e-3)
        if vl_select < best_val * (1.0 - min_delta_rel):
            best_val = vl_select
            bad_epochs = 0
            model.save(ckpt_path, epoch=epoch, best_val=best_val, optimizer=optimizer,
                       training_settings=training_settings)
            print("[Phase-3]: Current best model saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("[Phase-3]: Early stopping triggered.")
                break

    # Restore gradients to all parameters
    for param in model.parameters():
        param.requires_grad_(True)

    plot_losses(train_losses, val_losses)
    return model, train_losses, val_losses
