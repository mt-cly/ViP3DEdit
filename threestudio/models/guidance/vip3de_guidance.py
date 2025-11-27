from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import math
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from diffusers import AutoencoderKLTemporalDecoder
import concurrent.futures
from PIL import Image
from torchvision.io import read_video
import torch.nn.functional as F
import imageio
from einops import rearrange
from threestudio.utils.vip3de_utils import *

from torch.multiprocessing import Pool
from threestudio.utils.dge_utils import register_pivotal, register_batch_idx, register_cams, register_epipolar_constrains, register_extended_attention, register_normal_attention
import torchvision
import torch.multiprocessing as mp




def metric_depth_coloring(depth_map,
                          metric_range=(0.5, 8.0),  # 预设量程范围(单位：米)
                          gamma=0.6,
                          colormap=cv2.COLORMAP_JET):
    """
    改进版定标深度着色器
    :param depth_map: 输入深度图(浮点型矩阵)
    :param metric_range: 量程范围元组(min_depth, max_depth)
    :param gamma: 伽马校正参数(>1增强远处细节，<1增强近处细节)
    :param colormap: OpenCV色图类型
    """
    # 量程截断处理
    min_depth, max_depth = metric_range
    depth_clipped = np.clip(depth_map, min_depth, max_depth)

    # 固定量程归一化
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)

    # 伽马校正与量化
    depth_gamma = np.power(depth_normalized, gamma)
    depth_uint8 = np.uint8(depth_gamma * 255)

    # 创建自定义橙蓝渐变色图
    custom_colormap = np.zeros((256, 3), dtype=np.uint8)
    custom_colormap[:, 0] = np.linspace(120, 0, 256)  # 蓝色通道：120→0
    custom_colormap[:, 1] = np.linspace(70, 165, 256)  # 绿色通道：70→165
    custom_colormap[:, 2] = np.linspace(0, 255, 256)  # 红色通道：0→255

    # 应用色图
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    # 增强网格可见性
    # grid_size = 20  # 对应您图像中的网格密度
    # colored = cv2.addWeighted(colored, 0.9,
    #                           _create_grid(grid_size, colored.shape), 0.1, 0)

    return colored

def video_decoder(latent, svd_model, svd_filter):
    torch.cuda.empty_cache()
    svd_model.en_and_decode_n_samples_a_time = 6
    samples = svd_model.decode_first_stage(latent)
    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    samples = svd_filter(samples) * 255
    return samples


def edit_video(args):
    '''
    params:
        source_images: tensor with shape [f, 3, h, w]
        cams: [Camera()] with length f
        edit_first_frame: tensor with shape [3, h, w]
        num_clip_frames: int
    return:
        full_reconstruct: tensor with shape [f, 3, h, w]
        full_edit: tensor with shape [f, 3, h, w]
    '''
    source_images, cams, depths, masks, edit_first_frame, cfg, device = args
    source_images = source_images.to('cpu').to(device)
    depths = depths.to('cpu').to(device)
    masks = masks.to('cpu').to(device)
    edit_first_frame = edit_first_frame.to('cpu').to(device)
    print('loading instructp2p_vae and SVD')
    instructp2p_vae = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to(device).vae
    num_steps, num_clip_frames, inverse_alpha = cfg.num_steps, cfg.num_frames, cfg.inverse_alpha
    svd_model, svd_filter = load_model(
            cfg.svd_config,
            device,  # todo there are some bugs when assigning SVD(XT) to GPU other than CUDA:0
            cfg.num_frames,
            cfg.num_steps,
            verbose=False,
            min_scale=cfg.svd_min_scale,
            max_scale=cfg.svd_max_scale
        )


    print('begin edit video')
    full_reconstruct, full_edit = [], []
    num_imgs = len(source_images)
    num_extend_imgs = math.ceil((num_imgs-1) / (num_clip_frames-1)) * (num_clip_frames-1) + 1-num_imgs
    input_images = torch.stack([source_images[0], edit_first_frame])
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for idx_clip in range( (num_imgs+num_extend_imgs-1) // (num_clip_frames - 1)):
                # torch.cuda.empty_cache()
                video_clip = source_images[idx_clip * (num_clip_frames - 1): (idx_clip + 1) * (num_clip_frames - 1) + 1]
                depth_clip = depths[idx_clip * (num_clip_frames - 1): (idx_clip + 1) * (num_clip_frames - 1) + 1]
                camera_clip = cams[idx_clip * (num_clip_frames - 1): (idx_clip + 1) * (num_clip_frames - 1) + 1]
                mask_clip = masks[idx_clip * (num_clip_frames - 1): (idx_clip + 1) * (num_clip_frames - 1) + 1].repeat(1,3,1,1).bool()
                svd_model.sampler.guider.num_frames = len(video_clip) # update num_frames
                latent_clip = tensor_to_vae_latent(video_clip, instructp2p_vae)
                edit_first_frame_latent = tensor_to_vae_latent(input_images[1:2], instructp2p_vae)
                latent_clip_reconstruct, latent_clip_edit = edit_video_clip(svd_model, svd_filter,
                                                                            video_clip,
                                                                            depth_clip,
                                                                            latent_clip,
                                                                            camera_clip,
                                                                            edit_first_frame_latent,
                                                                            input_images,
                                                                            len(video_clip),
                                                                            num_steps=num_steps,
                                                                            inverse_alpha=inverse_alpha,
                                                                            device=device)
                # decode the latent
                video_clip_reconstruct = video_decoder(latent_clip_reconstruct, svd_model, svd_filter).to('cpu').to(device)
                video_clip_edit = video_decoder(latent_clip_edit, svd_model, svd_filter).to('cpu').to(device)
                # todo no mask region
                video_clip_edit[~mask_clip] = video_clip[~mask_clip]
                full_reconstruct.append(video_clip_reconstruct)
                full_edit.append(video_clip_edit)
                # prepare for next video clip
                input_images = torch.stack([video_clip[-1], video_clip_edit[-1].to('cpu').to(device)] )

    full_reconstruct = torch.cat(full_reconstruct, dim=0)[:num_imgs]
    full_edit = torch.cat(full_edit, dim=0)[:num_imgs]

    return full_reconstruct, full_edit


@threestudio.register("vip3de-guidance")
class DGEGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        diffusion_steps: int = 20
        use_sds: bool = False
        camera_batch_size: int = 5

        # svd_config: str = 'configs/svd.yaml'
        svd_config: str = 'configs/svd_xt.yaml'
        num_frames: int = 20
        num_steps: int = 25
        inverse_alpha: float = 0.1
        sampling_gap: int = 1
        svd_height: int = 576
        svd_width: int = 1024
        svd_min_scale: float = 1.0
        svd_max_scale: float = 2.5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f'Loading SVD and instructiPix2Pix ...')
        self.instructp2p_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                                      torch_dtype=torch.float16).to(self.device)
        # custom min/max_scale


        #
        #
        #
        #
        # threestudio.info(f"Loading InstructPix2Pix ...")
        #
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        #
        # pipe_kwargs = {
        #     "safety_checker": None,
        #     "feature_extractor": None,
        #     "requires_safety_checker": False,
        #     "torch_dtype": self.weights_dtype,
        #     "cache_dir": self.cfg.cache_dir,
        # }
        #
        # self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        #     self.cfg.ip2p_name_or_path, **pipe_kwargs
        # ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        #
        # if self.cfg.enable_memory_efficient_attention:
        #     if parse_version(torch.__version__) >= parse_version("2"):
        #         threestudio.info(
        #             "PyTorch2.0 uses memory efficient attention by default."
        #         )
        #     elif not is_xformers_available():
        #         threestudio.warn(
        #             "xformers is not available, memory efficient attention is not enabled."
        #         )
        #     else:
        #         self.pipe.enable_xformers_memory_efficient_attention()
        #
        # if self.cfg.enable_sequential_cpu_offload:
        #     self.pipe.enable_sequential_cpu_offload()
        #
        # if self.cfg.enable_attention_slicing:
        #     self.pipe.enable_attention_slicing(1)
        #
        # if self.cfg.enable_channels_last_format:
        #     self.pipe.unet.to(memory_format=torch.channels_last)
        #
        # # Create model
        # self.vae = self.pipe.vae.eval()
        # self.unet = self.pipe.unet.eval()
        #
        # for p in self.vae.parameters():
        #     p.requires_grad_(False)
        # for p in self.unet.parameters():
        #     p.requires_grad_(False)
        #
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        #
        # self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
        #     self.device
        # )
        #
        # self.grad_clip_val: Optional[float] = None
        #
        # threestudio.info(f"Loaded InstructPix2Pix!")
        # for _, module in self.unet.named_modules():
        #     if isinstance_str(module, "BasicTransformerBlock"):
        #         make_block_fn = make_dge_block
        #         module.__class__ = make_block_fn(module.__class__)
        #         # Something needed for older versions of diffusers
        #         if not hasattr(module, "use_ada_layer_norm_zero"):
        #             module.use_ada_layer_norm = False
        #             module.use_ada_layer_norm_zero = False
        #     register_extended_attention(self)

    
    @torch.amp.autocast('cuda', enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.amp.autocast('cuda', enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.amp.autocast('cuda', enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.amp.autocast('cuda', enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.amp.autocast('cuda', enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def use_normal_unet(self):
        print("use normal unet")
        register_normal_attention(self)


    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            cond_rgb: Float[Tensor, "B H W C"],
            depths,
            masks,
            prompt: str,
            gaussians=None,
            cams=None,
            render=None,
            pipe=None,
            background=None,
            seed=0,
            **kwargs,
    ):
        '''
        rgb: current rendered rgb [b h w 3]
        cond_rgb: original rendered image [b h w 3]
        return
        dict:{
                "edit_images": [b h w 3]
                "recons_images": [b 3h w 3]
                } 
        '''
        _, ORIG_H, ORIG_W, _ = rgb.shape
        SVD_H, SVD_W = self.cfg.svd_height, self.cfg.svd_width
        rgb = rgb.permute(0, 3, 1, 2)
        cond_rgb = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb = F.interpolate(cond_rgb, (SVD_H, SVD_W), mode='bilinear')
        rgb = F.interpolate(rgb, (SVD_H, SVD_W), mode='bilinear')
        masks = F.interpolate(masks, (SVD_H, SVD_W), mode='bilinear')
        depths = F.interpolate(depths.permute(0,3,1,2), (SVD_H, SVD_W), mode='bilinear').permute(0,2,3,1).contiguous()
        cond_rgb = (cond_rgb.clamp(0, 1)*255)# .type(torch.uint8)
        rgb = (rgb.clamp(0, 1)*255)# .type(torch.uint8)


        # editing
        if  len(rgb) > 25:  # split it into two left video and right video
            parallel = True
            keyframe_idx = len(rgb)//2
            edited_image = image_edit(rgb[keyframe_idx], prompt, self.instructp2p_pipeline, self.cfg.guidance_scale, self.cfg.condition_scale, seed)
            
            # Prepare arguments for parallel processing
            process_args = [
                # Left segment arguments
                (rgb[:keyframe_idx+1].flip(0),
                    cams[:keyframe_idx+1][::-1],
                    depths[:keyframe_idx+1].flip(0),
                    masks[:keyframe_idx+1].flip(0),
                    edited_image.clone(),
                    self.cfg,
                    "cuda:0"),
                # Right segment arguments
                (rgb[keyframe_idx:],
                    cams[keyframe_idx:],
                    depths[keyframe_idx:],
                    masks[keyframe_idx:],
                    edited_image,
                    self.cfg,
                    "cuda:1")  # 使用不同的GPU
            ]
            if parallel:
                # # Process both segments in parallel
                print('start edit video in parallel')
                mp.set_start_method('spawn', force=True)  # 确保使用spawn方法
                with Pool(processes=2) as pool:
                    results = pool.map(edit_video, process_args)
                recons_lvideo, edited_lvideo = results[0][0].to('cpu').to('cuda:0'), results[0][1].to('cpu').to('cuda:0')
                recons_rvideo, edited_rvideo = results[1][0].to('cpu').to('cuda:0'), results[1][1].to('cpu').to('cuda:0')
            else:
                # Process both segments in order
                print('start edit video in order')
                recons_lvideo, edited_lvideo = edit_video(process_args[0])
                recons_rvideo, edited_rvideo = edit_video(process_args[1])
                recons_lvideo, edited_lvideo = recons_lvideo.to('cpu').to('cuda:0'), edited_lvideo.to('cpu').to('cuda:0')
                recons_rvideo, edited_rvideo = recons_rvideo.to('cpu').to('cuda:0'), edited_rvideo.to('cpu').to('cuda:0')
            
            recons_video = torch.cat([recons_lvideo.flip(0), recons_rvideo[1:]], dim=0)
            edited_video = torch.cat([edited_lvideo.flip(0), edited_rvideo[1:]], dim=0)
        else:  # if video is no long, direct handle all frames without splitting left and right

            edited_image = image_edit(rgb[0],prompt, self.instructp2p_pipeline,
                                        self.cfg.guidance_scale, self.cfg.condition_scale, seed)
            process_args = (rgb, cams, depths, masks, edited_image, self.cfg, self.device)
            recons_video, edited_video = edit_video(process_args)

        rgb = F.interpolate(rgb, (ORIG_H, ORIG_W), mode="bilinear")
        recons_video = F.interpolate(recons_video, (ORIG_H, ORIG_W), mode="bilinear")
        edited_video = F.interpolate(edited_video, (ORIG_H, ORIG_W), mode="bilinear")

        return {"edit_images": edited_video.permute(0, 2, 3, 1) / 255,
                "construct_edit_images": torch.cat([rgb, recons_video, edited_video], axis=2).permute(0, 2, 3, 1) / 255}





















        assert cams is not None, "cams is required for dge guidance"
        batch_size, H, W, _ = rgb.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        RH, RW = height, width

        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)

        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(batch_size).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat(
            [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings],
            dim=0)  # [positive, negative, negative]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t, cams)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}



    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


