"""The pixel-space multi-view diffusion transformer.

PixWorld runs a Wan2.2 transformer **directly on pixels** -- there is no VAE and no
latent space anywhere in the model.  Two changes to the stock architecture make that work:

1. **A pixel-sized patch.**  The patch embedding is a ``(1, 32, 32)`` 3D convolution over
   RGB rather than a ``(1, 2, 2)`` convolution over 48-channel latents.  At 480x832 both
   produce the same 15x26 token grid, so the transformer's rotary positions, sequence
   length and compute are unchanged -- only what a token *is* changes.
2. **A decoder that maps tokens back to full resolution.**  Wan's own output head would
   have to emit ``3 * 32 * 32`` values per token; instead the tokens are handed to a
   U-Net (:class:`~detailer.DetailerUNet`) that injects them at its
   bottleneck, which already sits at ``H/32``.

The input is 13 channels per view::

    [ x_t 3 | raymap 6 | cond_rgb 3 | cond_mask 1 ]

* ``x_t`` is the noisy RGB of that view.
* ``raymap`` is the camera, encoded per pixel (:func:`cameras.create_raymaps`).
  An all-zero raymap means "no camera", which is how plain text-to-image rows are fed in.
* ``cond_rgb`` and ``cond_mask`` carry image conditioning: for image-to-3D the reference
  view is placed in ``cond_rgb`` with ``cond_mask = 1``, and every generated view has
  ``cond_mask = 0``.

The extra ten input channels are **zero-initialised**.

The model predicts ``x0`` directly (not velocity, not epsilon).

Views live on the transformer's temporal axis with ``p_t = 1``, so ``V`` views become
``V`` token frames and the rotary grid is ``(V, H/32, W/32)`` -- native, no interpolation.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import WanTransformer3DModel

from models.detailer import DetailerUNet

__all__ = ["MV_IN_CHANNELS", "WAN22_5B_PIXEL_CONFIG", "PixWorldMV",
           "cast_rope_like", "install_rope_precast"]

MV_IN_CHANNELS = 13     # [x_t 3 | raymap 6 | cond_rgb 3 | cond_mask 1] -- fixed layout

#: Wan2.2-TI2V-5B's transformer config with the two pixel modifications: the spatial patch
#: goes 2 -> 32 and the channel count 48 -> 3.  Everything else is stock, which is what
#: lets the pretrained blocks load unchanged.
WAN22_5B_PIXEL_CONFIG = dict(
    patch_size=(1, 32, 32), num_attention_heads=24, attention_head_dim=128,
    in_channels=3, out_channels=3, text_dim=4096, freq_dim=256, ffn_dim=14336,
    num_layers=30, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
    eps=1e-6, rope_max_seq_len=1024)


# ------------------------------------------------------------------ rope pre-cast --
def cast_rope_like(rotary_emb, dtype):
    """Cast every floating tensor in ``rotary_emb`` to ``dtype``, preserving structure.

    diffusers has shipped Wan's rotary embedding as a bare tensor, as a ``(cos, sin)``
    tuple, and as a complex tensor across versions.  All three pass through unharmed:
    complex tensors are not ``is_floating_point``, so they are left alone.
    """
    def _one(t):
        return (t.to(dtype) if torch.is_tensor(t) and t.is_floating_point() else t)
    if isinstance(rotary_emb, (tuple, list)):
        return type(rotary_emb)(_one(t) for t in rotary_emb)
    if isinstance(rotary_emb, dict):
        return {k: _one(v) for k, v in rotary_emb.items()}
    return _one(rotary_emb)


def install_rope_precast(model, dtype=torch.bfloat16, verbose=print):
    """Make ``transformer.rope`` emit ``dtype`` directly.  Idempotent.


    ``dtype`` must be the FSDP2 ``param_dtype`` the trunk is wrapped with.
    """
    t = model.transformer
    rope = t.rope
    if getattr(rope, "_pixworld_precast", None) is not None:
        return False
    orig = rope.forward

    def _fwd(*a, **kw):
        return cast_rope_like(orig(*a, **kw), dtype)

    rope.forward = _fwd
    rope._pixworld_precast = dtype
    if verbose:
        verbose(f"[pixworld] rope pre-cast to {dtype} installed")
    return True


# ------------------------------------------------------------------------- model --
class PixWorldMV(nn.Module):
    """Pixel-space multi-view Wan transformer with a U-Net pixel decoder.

    Args:
        config: transformer config; defaults to :data:`WAN22_5B_PIXEL_CONFIG`.

    Forward:
        ``hidden_states [B, 13, V, H, W]``, ``timestep [B]`` or ``[B, V]``,
        ``encoder_hidden_states [B, L, text_dim]``
        -> ``x0_hat [B, 3, V, H, W]``, plus a ``taps`` dict when ``return_taps=True``.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = dict(WAN22_5B_PIXEL_CONFIG if config is None else config)
        self.transformer = WanTransformer3DModel(**cfg)
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.patch_size = cfg["patch_size"]
        self.in_channels = cfg["in_channels"]
        self.gradient_checkpointing = False

        # Wan's own output head is never used: the pixel decoder produces the image.
        # Replacing it with an Identity keeps the attribute live for diffusers.
        self.transformer.proj_out = nn.Identity()

        # Widen the patch embedding 3 -> 13 input channels.  The extra columns are
        # EXACTLY zero.  The pretrained Wan patch embedding does not fit either
        # shape (it is 48-channel, patch 2), so it stays randomly initialised.
        old = self.transformer.patch_embedding
        new = nn.Conv3d(MV_IN_CHANNELS, self.inner_dim, kernel_size=old.kernel_size,
                        stride=old.stride).to(old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
            new.bias.copy_(old.bias)
        self.transformer.patch_embedding = new

        self.dip_head = DetailerUNet(ctx_dim=self.inner_dim)

    # ------------------------------------------------------------------ weights --
    @torch.no_grad()
    def load_wan_backbone(self, wan_state_dict):
        """Copy every shape-matching key from a stock Wan transformer state dict.

        The transformer blocks, the condition embedder and ``scale_shift_table`` all
        match and are copied.  ``patch_embedding`` does not (48 latent channels at patch
        2 versus 13 pixel channels at patch 32) and stays as initialised above.
        ``norm_out`` is parameter-free.

        Returns a dict with ``copied``, ``shape_mismatch`` and ``missing_in_src`` so the
        caller can log exactly what happened.
        """
        tgt = self.transformer.state_dict()
        copied, mismatch, missing = 0, [], 0
        for k, v in tgt.items():
            if k in wan_state_dict:
                if wan_state_dict[k].shape == v.shape:
                    tgt[k] = wan_state_dict[k]
                    copied += 1
                else:
                    mismatch.append(k)
            else:
                missing += 1
        self.transformer.load_state_dict(tgt, strict=True)
        return {"copied": copied, "shape_mismatch": mismatch, "missing_in_src": missing}

    # --------------------------------------------------------------- trainable --
    def set_trainable(self, n_edge_blocks=5, train_patchify=True):
        """Freeze everything, then unfreeze the set this model actually trains.

        That set is: the condition embedder, ``scale_shift_table``, the first and last
        ``n_edge_blocks`` transformer blocks, the patch embedding (optional), and the
        whole pixel decoder.  The middle transformer blocks stay frozen.

        Returns the set of trainable block indices.  ``n_edge_blocks`` is clamped to the
        number of blocks, so asking for more than exist trains all of them.
        """
        self.requires_grad_(False)
        t = self.transformer
        t.condition_embedder.requires_grad_(True)
        t.scale_shift_table.requires_grad_(True)
        n = len(t.blocks)
        ne = max(0, min(int(n_edge_blocks), n))
        train_idx = set(range(ne)) | set(range(max(0, n - ne), n))
        for i in train_idx:
            t.blocks[i].requires_grad_(True)
        if train_patchify:
            t.patch_embedding.requires_grad_(True)
        self.dip_head.requires_grad_(True)
        return train_idx

    def num_trainable(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, tot

    # ----------------------------------------------------------------- forward --
    def forward(self, hidden_states, timestep, encoder_hidden_states, return_taps=False):
        """See the class docstring for shapes.

        ``timestep`` is ``[B]`` when every view shares a noise level, or ``[B, V]`` for
        image-to-3D, where the reference view is pinned clean at ``t = 0``.  In the
        two-dimensional case the transformer's AdaLN conditions on the per-sample
        **max** (the generation level), while the decoder's FiLM sees the true per-view
        sigma -- so a clean anchor view is passed through instead of being "denoised".
        Which view is the anchor is also carried into the transformer by ``cond_mask``.

        With ``return_taps=True`` the second return value is
        ``{"ctx", "x_t", "sig"}``, all folded view-major to ``[B*V, ...]`` in the same
        ``b*V + v`` order as ``cameras.flatten(0, 1)``.  These are the inputs to
        :func:`gs_lift.gs_lift`.  The taps carry no decoder tensor.
        """
        t = self.transformer
        B, C, V, H, W = hidden_states.shape
        assert C == MV_IN_CHANNELS, \
            f"expected {MV_IN_CHANNELS}-ch [x|raymap|cond|mask], got C={C}"
        _, p_h, p_w = self.patch_size
        pph, ppw = H // p_h, W // p_w

        if timestep.dim() == 2:
            ts_dit = timestep.amax(dim=1)                     # [B]
            sig_f = (timestep.float() / 1000.0).reshape(-1)   # [B*V], b*V+v order
        else:
            ts_dit = timestep
            sig_f = (timestep.float() / 1000.0).repeat_interleave(V)

        rotary_emb = t.rope(hidden_states)
        x = t.patch_embedding(hidden_states).flatten(2).transpose(1, 2)  # [B,V*pph*ppw,dim]
        # Only positional args: WanTimeTextImageEmbedding.forward has the
        # `timestep_seq_len` kwarg in some diffusers versions and not others.  It is
        # None here either way, so omitting it keeps this version-agnostic.
        temb, timestep_proj, enc_hs, _ = t.condition_embedder(
            ts_dit, encoder_hidden_states, None)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        for block in t.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, enc_hs, timestep_proj, rotary_emb, use_reentrant=False)
            else:
                x = block(x, enc_hs, timestep_proj, rotary_emb)

        # Wan's native final layer: AdaLN over the parameter-free norm_out.
        shift, scale = (t.scale_shift_table.to(temb.device)
                        + temb.unsqueeze(1)).chunk(2, dim=1)
        x = (t.norm_out(x.float()) * (1 + scale) + shift).type_as(hidden_states)

        # Sequence order is (V, pph, ppw), so a reshape gives per-view context already
        # at the native /32 grid -- fed straight to the decoder with no interpolation.
        ctx = x.transpose(1, 2).reshape(B, self.inner_dim, V, pph, ppw)
        ctx = ctx.permute(0, 2, 1, 3, 4).reshape(B * V, self.inner_dim, pph, ppw)
        x_t = hidden_states[:, :3].permute(0, 2, 1, 3, 4).reshape(B * V, 3, H, W)

        out = self.dip_head(x_t, ctx, sig_f)                  # [B*V, 3, H, W]
        out = out.reshape(B, V, 3, H, W).permute(0, 2, 1, 3, 4)
        if return_taps:
            return out, {"ctx": ctx, "x_t": x_t, "sig": sig_f}
        return out
