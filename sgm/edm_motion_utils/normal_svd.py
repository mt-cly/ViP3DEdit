from functools import partial
from typing import List, Optional, Union

import torch
from einops import rearrange, repeat
from sgm.modules.attention import checkpoint, exists
from sgm.modules.diffusionmodules.util import timestep_embedding
from torchvision.utils import save_image

#
# ### VideoUnet #####
# def forward_VideoUnet(
#         self,
#         x: torch.Tensor,
#         timesteps: torch.Tensor,
#         context: Optional[torch.Tensor] = None,
#         y: Optional[torch.Tensor] = None,
#         time_context: Optional[torch.Tensor] = None,
#         num_video_frames: Optional[int] = None,
#         image_only_indicator: Optional[torch.Tensor] = None
# ):
#     assert (y is not None) == (
#             self.num_classes is not None
#     ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
#     hs = []
#     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#     emb = self.time_embed(t_emb)
#
#     if self.num_classes is not None:
#         assert y.shape[0] == x.shape[0]
#         emb = emb + self.label_emb(y)
#
#     h = x
#     for module in self.input_blocks:
#         h = module(
#             h,
#             emb,
#             context=context,
#             image_only_indicator=image_only_indicator,
#             time_context=time_context,
#             num_video_frames=num_video_frames,
#         )
#         hs.append(h)
#     h = self.middle_block(
#         h,
#         emb,
#         context=context,
#         image_only_indicator=image_only_indicator,
#         time_context=time_context,
#         num_video_frames=num_video_frames,
#     )
#     for module in self.output_blocks:
#         h = torch.cat([h, hs.pop()], dim=1)
#         h = module(
#             h,
#             emb,
#             context=context,
#             image_only_indicator=image_only_indicator,
#             time_context=time_context,
#             num_video_frames=num_video_frames,
#         )
#     h = h.type(x.dtype)
#     return self.out(h)

# ============================================
# ============================================
# temporal alignment

### VideoTransformerBlock #####
def NormalVideoTransformerBlock_forward(self, x, context, timesteps):
    if self.checkpoint:
        return checkpoint(self._forward, x, context, timesteps)
    else:
        return self._forward(x, context, timesteps=timesteps)

def NormalVideoTransformerBlock__forward(self, x, context=None, timesteps=None):
    assert self.timesteps or timesteps
    assert not (self.timesteps and timesteps) or self.timesteps == timesteps
    timesteps = self.timesteps or timesteps
    B, S, C = x.shape
    x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

    if self.ff_in:
        x_skip = x
        x = self.ff_in(self.norm_in(x))
        if self.is_res:
            x += x_skip

    if self.disable_self_attn:
        x_ = self.attn1(self.norm1(x), context=context)
        x = x_ + x
    else:
        # original self-attn
        x = self.attn1(self.norm1(x)) + x

    if self.attn2 is not None:
        if self.switch_temporal_ca_to_sa:
            x = self.attn2(self.norm2(x)) + x
        else:
            # original cross-attn
            x = self.attn2(self.norm2(x), context=context) + x

    x_skip = x
    x = self.ff(self.norm3(x))
    if self.is_res:
        x += x_skip

    x = rearrange(
        x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
    )
    return x

from inspect import isfunction
import math
from packaging import version
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    import logging
    logpy = logging.getLogger(__name__)
    logpy.warn("no module 'xformers'. Processing without...")



# ==============================================
# ==============================================
# align spatial

def NormalBasicTransformerBlock__forward(
    self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
):
    # original attn1
    x = (
        self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None,
            additional_tokens=additional_tokens,
            n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
            if not self.disable_self_attn
            else 0,
        )
        + x
    )

    # =============== calculate the correspondance
    # x_uc_source, x_uc_edit, x_c_source, x_c_edit = x.chunk(4)
    # num_f, num_s, dim = x_uc_source.shape
    # if hasattr(self, "epipolar_constrains"):
    #     sim = torch.einsum("f s c, F S c-> f F s S",
    #                        x_c_source / x_c_source.norm(dim=-1, keepdim=True),
    #                        x_c_source[0:1] / x_c_source[0:1].norm(dim=-1, keepdim=True))
    #     sim = rearrange(sim, "f F s S -> (f s) (F S)")
    #     print(f'using {num_s}')
    #     # idx1_epipolar =  rearrange(self.epipolar_constrains[num_s], "f F s S -> (f s) (F S)")
    #     idx1_epipolar = self.epipolar_constrains[num_s]
    #     valid_mask = rearrange(idx1_epipolar.sum(-1) != num_s, '(f s)-> f s', f=num_f, s=num_s)
    #     idx1 = rearrange(sim.max(dim=-1)[1], " (f s)-> f s", f=num_f, s=num_s)[..., None].repeat(1, 1, dim)
    #     x_uc_source[valid_mask] = x_uc_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]
    #     x_c_source[valid_mask] = x_c_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]
    #     x_uc_edit[valid_mask] = x_uc_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]
    #     x_c_edit[valid_mask] = x_c_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]
    # x = torch.cat((x_uc_source, x_uc_edit, x_c_source, x_c_edit), dim=0)
    # ================================================

    # original attn2
    x = (
        self.attn2(
            self.norm2(x), context=context, additional_tokens=additional_tokens
        )
        + x
    )



    x = self.ff(self.norm3(x)) + x
    return x