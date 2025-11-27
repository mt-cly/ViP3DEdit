from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

class FeatureSaveMatch:
    def __init__(self):
        self.stacks = []

    def save_features(self, features):
        self.stacks.append(features)

    def show_similarity(self, videos):
        '''
        calculate the semantic correspondance in source videos
        videos: [1, f, h, w, 3]

        '''
        videos = videos[0]

        feats = []
        for idx, feat in enumerate(self.stacks):
            scale = int((feat.shape[1]//16//9)**0.5)
            if scale not in [4, 2, 1]:
                continue
            if idx < 6:
                continue
            feat = rearrange(feat, '(n b) (h w) c -> n b c h w', n=2, h=scale*9, w=scale*16)[0]
            feat = F.interpolate(feat, (72, 128), mode='bilinear')
            feats.append(rearrange(feat, 'b c h w -> b h w c'))
        feats = torch.cat(feats, dim=-1)

        assert videos.shape[0] == len(feats)
        # similarity = torch.einsum('qhwc,fjkc->qhwfjk',feats[0:1], feats)[0] # [h w f h w]
        # _, _, _, h, w = similarity.shape
        # videos = F.interpolate(videos.float(), (h, w), mode='bilinear')
        # pnt_reference = [0.5, 0.5]
        # simi_map = similarity[int(pnt_reference[0]*h), int(pnt_reference[1]*w)]
        # simi_map = simi_map / torch.amax(simi_map, (1, 2))[:, None, None]
        # save_map = videos/255 * 0.0 + simi_map[:,None]* 1.
        # save_map = rearrange(save_map, 'f c h w -> c h (f w)')
        # save_image(save_map, 'correspondace.png')


        _, h, w, _ = feats.shape
        videos = F.interpolate(videos.float(), (h, w), mode='bilinear')
        pnt_reference = [0.6, 0.4]
        anchor_feat = feats[0:1, int(pnt_reference[0]*h), int(pnt_reference[1]*w)]
        simi = torch.exp(-abs(anchor_feat[:,None,None] - feats).mean(-1))
        simi_map = (simi == torch.amax(simi, (1, 2))[:, None, None]).float()
        save_map = videos/255 * 0.2 + simi_map[:,None]* 0.8
        save_map = rearrange(save_map, 'f c h w -> c h (f w)')
        save_image(save_map, 'correspondace.png')

        pass


def FeatureExtractBasicTransformerBlock__forward(
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


    # original attn2
    x = (
        self.attn2(
            self.norm2(x), context=context, additional_tokens=additional_tokens
        )
        + x
    )


    x = self.ff(self.norm3(x)) + x

    self.feautreSaveMatch.save_features(x.clone())
    return x