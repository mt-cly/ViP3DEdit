from functools import partial
from typing import List, Optional, Union

import torch
import torchvision.io
from einops import rearrange, repeat
from sgm.modules.attention import checkpoint, exists
from sgm.modules.diffusionmodules.util import timestep_embedding
from torchvision.utils import save_image
from inspect import isfunction
import matplotlib.pyplot as plt
import math
import torchvision
from packaging import version
import random
import numpy as np
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    import logging
    logpy = logging.getLogger(__name__)
    logpy.warn("no module 'xformers'. Processing without...")






def visualize_correspondence(correspondence, epipolar_mask, images, save_path="correspondence_visualization.png"):
    from skimage.transform import resize
    """
    从第二张图像中随机选择一个点，可视化其 correspondence 并保存结果。

    参数:
        correspondence (np.ndarray): 形状为 [fhw] 的 correspondence 张量。
        epipolar_mask (np.array): 形状为[fhw, hw]的mask
        images (list of np.ndarray): 形状为 [f, h, w, 3] 的图像列表。
        save_path (str): 可视化结果的保存路径。
    """
    # 检查输入形状是否匹配
    num_f, img_h, img_w,_ = images.shape
    scale = int((num_f * img_h * img_w // correspondence.shape[-1])**0.5)
    correspondence = correspondence.reshape(num_f, img_h//scale, img_w //scale)
    epipolar_mask = epipolar_mask.reshape(num_f, img_h//scale, img_w //scale, img_h//scale, img_w //scale).astype(np.float32)
    assert correspondence.shape[0] == len(images), "correspondence 和 images 的第一维度不匹配"

    # 获取第二张图像.
    ref_img_idx = 15 #  random.randint(0, len(images) - 1)
    second_image = images[ref_img_idx]

    # 随机选择一个点 (h, w)
    h, w = correspondence.shape[1], correspondence.shape[2]
    random_h = random.randint(0, h - 1)
    random_w = random.randint(0, w - 1)

    # 获取该点的 correspondence 索引
    correspondence_index = correspondence[ref_img_idx, random_h, random_w]
    print(correspondence_index)
    # 找到对应点的位置
    target_f = correspondence_index // (h * w)
    target_h = (correspondence_index % (h * w)) // w
    target_w = correspondence_index % w

    # 获取对应图像
    target_image = images[target_f]

    # 获取极线掩码
    epipolar_mask_selected = epipolar_mask[ref_img_idx, random_h, random_w]

    # 可视化
    plt.figure(figsize=(10, 5))

    # 显示第二张图像和随机点
    plt.subplot(1, 2, 1)
    plt.imshow(second_image)
    plt.scatter(random_w* scale, random_h* scale, c='red', marker='o', s=50)  # 标记随机点
    plt.title(f"Second Image (Point: [{random_h}, {random_w}])")

    # 显示对应图像和对应点
    target_image = target_image * 0.5 + resize(epipolar_mask_selected, (img_h, img_w), order=1)[...,None] * 0.5
    plt.subplot(1, 2, 2)
    plt.imshow(target_image)
    plt.scatter(target_w* scale, target_h* scale, c='blue', marker='o', s=50)  # 标记对应点
    plt.title(f"Target Image (Point: [{target_h}, {target_w}])")

    # 保存结果
    plt.savefig(save_path)
    plt.close()
    print(f"可视化结果已保存到: {save_path}")



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
def ReplacementVideoTransformerBlock_forward(self, x, context, timesteps):
    if self.checkpoint:
        return checkpoint(self._forward, x, context, timesteps)
    else:
        return self._forward(x, context, timesteps=timesteps)

def ReplacementVideoTransformerBlock__forward(self, x, context=None, timesteps=None):
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
        if False:
            # original self-attn
            x = self.attn1(self.norm1(x)) + x
        else:
            # rewrite to support self-attn replacement from source to edit
            num_uc_c = 2  # uc,c
            x_split = rearrange(x, "(n b s) t c -> b (n s) t c", n=num_uc_c, b=2, s=S) # b=2 with reconstruction and editting
            x_source = x_split[0]
            x_edit = x_split[1]
            x_source, attn_map = self.attn1(self.norm1(x_source), if_return_attnmap=True)
            attn_map = attn_map if hasattr(self, 'attn_map_overriding') else None
            x_edit = self.attn1(self.norm1(x_edit), given_attnmap=attn_map)
            x_source_edit = torch.stack([x_source, x_edit])
            x_source_edit = rearrange(x_source_edit, 'b (n s) t c -> (n b s) t c', b=2, n=num_uc_c, s=S)
            x = x_source_edit + x

    if self.attn2 is not None:
        if self.switch_temporal_ca_to_sa:
            x = self.attn2(self.norm2(x)) + x
        else:
            if True:
                # original cross-attn
                x = self.attn2(self.norm2(x), context=context) + x
            else:
                # rewrite to support cross-attn replacement from source to edit
                num_c_uc = 2 # uc,c
                x_split = rearrange(x, "(n b s) t c -> b (n s) t c", n=num_c_uc, b=2, s=S)
                x_source = x_split[0]
                x_edit = x_split[1]
                cond_split = rearrange(context, "(n b s) t c ->  b (n s) t c", n=num_c_uc, b=2, s=S) # [[uc-source, uc-edit], [c-source, c-edit]]
                cond_source = cond_split[0]
                cond_edit = cond_split[1]
                x_source, attn_map = self.attn2(self.norm2(x_source), context=cond_source, if_return_attnmap=True)
                attn_map = attn_map if hasattr(self, 'attn_map_overriding') else None
                x_edit = self.attn2(self.norm2(x_edit), context=cond_edit, given_attnmap=attn_map)
                x_source_edit = torch.stack([x_source, x_edit])
                x_source_edit = rearrange(x_source_edit, 'b (n s) t c -> (n b s) t c', b=2, n=num_c_uc, s=S)
                x = x_source_edit + x

    x_skip = x
    x = self.ff(self.norm3(x))
    if self.is_res:
        x += x_skip

    x = rearrange(
        x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
    )
    return x




# ==============================================
# ==============================================
# align spatial

def ReplacementBasicTransformerBlock__forward(
    self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
):
    if False:
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
    else:

        # rewrite to support self-attn replacement from source to edit
        x_uc_source, x_uc_edit, x_c_source, x_c_edit = x.chunk(4)
        num_f, num_s, dim = x_uc_source.shape
        x_source = torch.cat([x_uc_source, x_c_source], dim=0)
        x_edit = torch.cat([x_uc_edit, x_c_edit], dim=0)
        context_uc_source, context_uc_edit, context_c_source, context_c_edit = context.chunk(4)
        context_source = torch.cat([context_uc_source, context_c_source], dim=0)
        context_edit = torch.cat([context_uc_edit, context_c_edit], dim=0)
        x_source, attn_map = self.attn1(
                self.norm1(x_source),
                context=context_source if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn else 0,
                if_return_attnmap=True
            )

        attn_map = attn_map if hasattr(self, 'attn_map_overriding') else None
        x_edit = self.attn1(
                self.norm1(x_edit),
                context=context_edit if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn else 0,
                given_attnmap=attn_map
            )
        x_uc_source, x_c_source = x_source.chunk(2)
        x_uc_edit, x_c_edit = x_edit.chunk(2)
        # =============== calculate the correspondance
        if hasattr(self, "constrains"):
            sim = torch.einsum("f s c, F S c-> f F s S",
                               x_source.chunk(2)[1] / x_source.chunk(2)[1].norm(dim=-1, keepdim=True),
                               x_source.chunk(2)[1][0:1] / x_source.chunk(2)[1][0:1].norm(dim=-1, keepdim=True))
            sim = rearrange(sim, "f F s S -> (f s) (F S)")

            idx1_constrains = self.constrains[num_s]
            valid_mask = rearrange(idx1_constrains.sum(-1) != num_s, '(f s)-> f s', f=num_f, s=num_s)
            idx1 = rearrange(sim.max(dim=-1)[1], " (f s)-> f s", f=num_f, s=num_s)[..., None].repeat(1, 1, dim)
            x_c_source[valid_mask] = x_c_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]
            x_c_edit[valid_mask] = x_c_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]
            # x_uc_edit[valid_mask] = x_uc_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)[valid_mask]

            # imgs = torch.stack([torchvision.io.read_image(
            #     '/data0/liyi/code/DGE/edit_cache_svd_xt/bear_smooth_v7_Turn_it_into_a_grizzly_bear./origin_render/00%02d.png' % (
            #                 17 + i)) for i in range(25)]).permute(0, 2, 3, 1).cpu().numpy()
            # visualize_correspondence(idx1[..., 0].flatten().cpu().numpy(),
            #                          idx1_constrains.cpu().numpy(),
            #                          imgs/255,
            #                          save_path="vis_correspondance.png")
        # =============================================
        x = torch.cat((x_uc_source, x_uc_edit, x_c_source, x_c_edit), dim=0) + x

    if True: # original attn2
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
    else: # source attn2 qk -> edit attn2 qk
        # rewrite to support cross-attn replacement from source to edit
        x_uc_source, x_uc_edit, x_c_source, x_c_edit = x.chunk(4)
        x_source = torch.cat([x_uc_source, x_c_source], dim=0)
        x_edit = torch.cat([x_uc_edit, x_c_edit], dim=0)
        context_uc_source, context_uc_edit, context_c_source, context_c_edit = context.chunk(4)
        context_source = torch.cat([context_uc_source, context_c_source], dim=0)
        context_edit = torch.cat([context_uc_edit, context_c_edit], dim=0)
        x_source, attn_map = self.attn2(
            self.norm2(x_source),
            context=context_source,
            additional_tokens=additional_tokens,
            if_return_attnmap=True
        )
        attn_map = attn_map if hasattr(self, 'attn_map_overriding') else None
        x_edit = self.attn2(
            self.norm2(x_edit),
            context=context_edit,
            additional_tokens=additional_tokens,
            given_attnmap=attn_map if hasattr(self, 'attn_map_overriding') else None
        )
        x_uc_source, x_c_source = x_source.chunk(2)
        x_uc_edit, x_c_edit = x_edit.chunk(2)
        # =============== calculate the correspondance
        # if hasattr(self, "constrains"):
        #     sim = torch.einsum("f s c, F S c-> f F s S",
        #                        x_source.chunk(2)[1] / x_source.chunk(2)[1].norm(dim=-1, keepdim=True),
        #                        x_source.chunk(2)[1][0:1] / x_source.chunk(2)[1][0:1].norm(dim=-1, keepdim=True))
        #     sim = rearrange(sim, "f F s S -> (f s) (F S)")
        #     print(f'using {num_s}')
        #     idx1_constrains = rearrange(self.constrains[num_s], "f F s S -> (f s) (F S)")
        #     idx1_constrains[idx1_constrains.sum(-1) == num_s] = False
        #     sim[idx1_constrains] = -1
        #     idx1 = rearrange(sim.max(dim=-1)[1], " (f s)-> f s", f=num_f, s=num_s)[..., None].repeat(1, 1, dim)
            # x_uc_source = x_uc_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
            # x_c_source = x_c_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
            # x_uc_edit = x_uc_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
            # x_c_edit = x_c_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
        # =============================================
        x = torch.cat((x_uc_source, x_uc_edit, x_c_source, x_c_edit), dim=0) + x

    x = self.ff(self.norm3(x)) + x

    # =============== calculate the correspondance
    # if hasattr(self, "constrains"):
    #     x_uc_source, x_uc_edit, x_c_source, x_c_edit = x.chunk(4)
    #     sim = torch.einsum("f s c, F S c-> f F s S",
    #                        x_c_source / x_c_source.norm(dim=-1, keepdim=True),
    #                        x_c_source[0:1] / x_c_source[0:1].norm(dim=-1, keepdim=True))
    #     sim = rearrange(sim, "f F s S -> (f s) (F S)")
    #     print(f'using {num_s}')
    #     idx1_constrains = rearrange(self.constrains[num_s], "f F s S -> (f s) (F S)")
    #     idx1_constrains[idx1_constrains.sum(-1) == num_s] = False
    #     sim[idx1_constrains] = -1
    #     idx1 = rearrange(sim.max(dim=-1)[1], " (f s)-> f s", f=num_f, s=num_s)[..., None].repeat(1, 1, dim)
    #     x_uc_source = x_uc_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
    #     x_c_source = x_c_source[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
    #     # x_uc_edit = x_uc_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
    #     # x_c_edit = x_c_edit[0:1].repeat(num_f, 1, 1).gather(dim=1, index=idx1)
    #
    #     x = torch.cat((x_uc_source, x_uc_edit, x_c_source, x_c_edit), dim=0)
    #
    #     imgs = torch.stack([torchvision.io.read_image('/data0/liyi/code/DGE/edit_cache_svd_xt/bear_smooth_v7_Turn_it_into_a_grizzly_bear./origin_render/00%02d.png'%(17+i)) for i in range(25)]).permute(0,2,3,1).cpu().numpy()
    #     visualize_correspondence(idx1[...,0].flatten().cpu().numpy(), self.constrains[num_s].flatten(0,2).cpu().numpy(), imgs, save_path="vis_correspondance.png" )
    # =============================================
    return x



# rewrite the forward of MemoryEfficientCrossAttention
# supporting return qk attn map and specify the qk attn map
def extend_MemoryEfficientCrossAttention(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        if_return_attnmap=False,
        given_attnmap=None
):
    def default(val, d):
        if exists(val):
            return val
        return d() if isfunction(d) else d

    if additional_tokens is not None:
        # get the number of masked tokens at the beginning of the output sequence
        n_tokens_to_mask = additional_tokens.shape[1]
        # add additional token
        x = torch.cat([additional_tokens, x], dim=1)
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    if n_times_crossframe_attn_in_self:
        # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
        assert x.shape[0] % n_times_crossframe_attn_in_self == 0
        # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
        k = repeat(
            k[::n_times_crossframe_attn_in_self],
            "b ... -> (b n) ...",
            n=n_times_crossframe_attn_in_self,
        )
        v = repeat(
            v[::n_times_crossframe_attn_in_self],
            "b ... -> (b n) ...",
            n=n_times_crossframe_attn_in_self,
        )

    b, _, _ = q.shape
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, t.shape[1], self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * self.heads, t.shape[1], self.dim_head)
        .contiguous(),
        (q, k, v),
    )

    not_align_uc = True

    # actually compute the attention, what we cannot get enough of
    if version.parse(xformers.__version__) >= version.parse("0.0.21"):
        # NOTE: workaround for
        # https://github.com/facebookresearch/xformers/issues/845
        max_bs = 32768
        N = q.shape[0]
        n_batches = math.ceil(N / max_bs)
        out = list()
        for i_batch in range(n_batches):
            batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
            if given_attnmap is not None:
                new_q = given_attnmap[0][batch]
                new_k = given_attnmap[1][batch]
                if not_align_uc:
                    new_q = torch.cat([q[batch].chunk(2)[0], given_attnmap[0][batch].chunk(2)[1]])
                    new_k = torch.cat([k[batch].chunk(2)[0], given_attnmap[1][batch].chunk(2)[1]])
                new_v = v[batch]
                out.append(
                    xformers.ops.memory_efficient_attention(
                        new_q,
                        new_k,
                        new_v,
                        op=self.attention_op,
                    )
                )
            else:
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        op=self.attention_op,
                    )
                )
        out = torch.cat(out, 0)
    else:
        if given_attnmap is not None:
            new_q = given_attnmap[0]
            new_k = given_attnmap[1]
            if not_align_uc:
                new_q = torch.cat([q.chunk(2)[0], given_attnmap[0].chunk(2)[1]])
                new_k = torch.cat([k.chunk(2)[0], given_attnmap[1].chunk(2)[1]])
            new_v = v
            out = xformers.ops.memory_efficient_attention(
                new_q,
                new_k,
                new_v, op=self.attention_op
            )
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, op=self.attention_op
            )

    # TODO: Use this directly in the attention operation, as a bias
    if exists(mask):
        raise NotImplementedError
    out = (
        out.unsqueeze(0)
        .reshape(b, self.heads, out.shape[1], self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, out.shape[1], self.heads * self.dim_head)
    )
    if additional_tokens is not None:
        # remove additional token
        out = out[:, n_tokens_to_mask:]

    if if_return_attnmap:
        return self.to_out(out), (q, k, v)
    else:
        return self.to_out(out)




