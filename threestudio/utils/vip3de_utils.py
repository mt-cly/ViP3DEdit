import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import torchvision.utils
from torchvision.io import read_image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from threestudio.utils.dge_utils import compute_epipolar_constrains, compute_depth_constrains, register_epipolar_constrains
import random
def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = t / 127.5 - 1
    with torch.no_grad():
        latents = vae.encode(t).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    return latents


def image_edit(image, prompt, model, guidance_scale, image_guidance_scale=1.5, seed=0):
    '''
    Args:
       image: tensor(3, h, w) in [0, 255]
       prompt: str
       model: instructPix2Pix
    return:
       edited image: tensor(3, h, w) in [0, 255]
    '''
    # todo: using instructp2p or preset image
    # if True:
    #     return image_edit_preset(image, prompt, model, guidance_scale, image_guidance_scale, seed)

    generator = torch.Generator("cuda").manual_seed(seed)
    image = image.type(torch.uint8).cpu().numpy().transpose((1, 2, 0))
    h, w, _ = image.shape
    num_inference_steps = 20
    image = Image.fromarray(image)#.convert("RGB")
    edited_image = model(
        prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    edited_image = edited_image.resize((w, h))
    edited_image = np.array(edited_image).transpose(2, 0, 1)

    edited_image = torch.from_numpy(edited_image).to(model.device)

    return edited_image

def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
    min_scale: float = 1.0,
    max_scale: float = 2.5
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config['params']['min_scale'] = min_scale
    config.model.params.sampler_config.params.guider_config['params']['max_scale'] = max_scale
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model)
        model = model.to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter

def create_batch(model, input_images, motion_bucket_id, fps_id, cond_aug, num_frames, device):
    from scripts.sampling.simple_video_sample import get_batch,get_unique_embedder_keys_from_conditioner
    images = input_images / 127.5 - 1
    c_list, uc_list = [], []
    c_combine, uc_combine = {}, {}
    cond_aug_noise = cond_aug * torch.randn_like(images[0])
    for image in images:
        image = image[None]
        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug_noise

        _, _ , H, W = image.shape
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(model.conditioner),
            value_dict,
            [1, num_frames],
            T=num_frames,
            device=device,
        )
        c, uc = model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
            c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

        c_list.append(c)
        uc_list.append(uc)

    for key in c_list[0].keys():
        c_combine[key] = torch.cat([c[key] for c in c_list])
        uc_combine[key] = torch.cat([uc[key] for uc in uc_list])

    return c_combine, uc_combine



def palette(img):
    '''
    img: [bs, c, h, w]
    '''
    img = (img*8)+128
    img = img/255
    return img
def brokenV1_energy(inversed_noise, inverse_alpha):
    noise = inversed_noise * (inverse_alpha ** 0.5) + torch.randn_like(inversed_noise) * (
            (1 - inverse_alpha) ** 0.5)

    # [save_image(palette(inversed_noise)[i], f'inversed_noise_{i}.png') for i in range(25)]
    # [save_image(palette(noise)[i], f'noise_{i}.png') for i in range(25)]
    # save_image(palette(noise)[0], 'noise.png')
    return noise

def brokenV2_lowpass(inversed_noise, inverse_alpha):
    from torchvision.transforms import GaussianBlur
    filter = GaussianBlur((5, 5), 1)
    noise = filter(inversed_noise)
    noise = noise/noise.std()

    return noise

def brokenV3_reduce(inversed_noise, inverse_alpha):
    noise = inversed_noise * 0.2 + torch.randn_like(inversed_noise) * 0.8
    return noise

def brokenV4_subtemporalmean(inversed_noise, inverse_alpha):
    noise = inversed_noise - inversed_noise.mean(0, keepdims=True)
    return noise

def brokenV5_subspatialmean(inversed_noise, inverse_alpha):
    noise = inversed_noise - inversed_noise.mean((1,2,3), keepdims=True)
    return noise


def edit_video_clip(model,
                  filter,
                  video,
                  depths,
                  latent,
                  cams,
                  edit_first_frame_latent,
                  input_images,
                  num_frames: Optional[int] = None,  # 21 for SV3D
                  num_steps: Optional[int] = None,
                  inverse_alpha: float = 0.1,
                  version: str = "svd",
                  fps_id: int = 6,
                  motion_bucket_id: int = 127,
                  cond_aug: float = 0.02,
                  seed: int = 26,
                  decoding_t: int = 14, # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
                  device: str = "cuda",
                  output_folder: Optional[str] = 'edit_video_vis',
                  img_name: str = "cond_img",
                  verbose: Optional[bool] = False,
                  ):
    '''
    args:
        model: SVD
        filter:
        latent: [bf, c, h, w]
        cameras: [bf] Camera()
        edit_first_frame_latent: [1, 4, h, w] used for align mean/std of output
        input_images: [2, 3, h, w] condition, which should be [0, 255]
        num_frames: int
        num_steps: int
        device: device
    return:
        samples_reconstruct:
        samples_edit:
    '''

    output_device = device
    device = model.device
    latent = latent.to('cpu').to(device)
    edit_first_frame_latent = edit_first_frame_latent.to('cpu').to(device)
    input_images = input_images.to('cpu').to(device)
    bs, _, img_H, img_W = input_images.shape

    # 1. epipolar  constraint
    # epipolar_constrains = {}
    # for down_sample_factor in [1, 2, 4, 8]:
    #     H = img_H // 8 // down_sample_factor
    #     W = img_W // 8 // down_sample_factor
    #     epipolar_constrains[H * W] = []
    #     for cam, depth in zip(cams, depths):
    #         cam_epipolar_constrains = []
    #         for key_cam, key_depth in zip(cams[0:1], depths[0:1]):
    #             if cam == key_cam:
    #                 cam_epipolar_constrains.append((torch.eye(H*W)<0.5).cuda())
    #             else:
    #                 cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam,  cam,  current_H=H, current_W=W))
    #         epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
    #     epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0).to("cpu").to(device)
    # 1. depth constraint
    depth_constraints = {}
    for down_sample_factor in [1, 2, 4, 8]:
        H = img_H // 8 // down_sample_factor
        W = img_W // 8 // down_sample_factor
        depth_constraints[H * W] = compute_depth_constrains(cams[0], depths[0], cams, depths, current_H=H, current_W=W)
    # depth_constraints[img_H * img_W] = compute_depth_constrains(cams[0], depths[0], cams, depths, current_H=img_H, current_W=img_W)
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
        ref_img_idx =  random.randint(0, len(images) - 1)
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
        target_image = images[0]

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

    video = video.permute(0,2,3,1)
    # visualize_correspondence(depth_constraints[6144][:,0].cpu().numpy()*0, depth_constraints[6144].cpu().numpy(), video.cpu().numpy()/255, save_path="correspondence_visualization.png")

    # 2. inverse and denoising
    additional_model_inputs = {}
    additional_model_inputs["image_only_indicator"] = torch.zeros(2*bs, num_frames).to(device)  # 2 for (uc,c)
    additional_model_inputs["num_video_frames"] = num_frames

    assert input_images.shape[1] == 3
    F = 8
    C = 4
    c, uc = create_batch(model, input_images, motion_bucket_id, fps_id, cond_aug, num_frames, device)
    # visualization
    os.makedirs(output_folder, exist_ok=True)
    for batch_id in range(len(input_images)):
        Image.fromarray(input_images[batch_id].type(torch.uint8).cpu().numpy().transpose(1, 2, 0)).save(os.path.join(output_folder, f"{img_name}_{batch_id}.png"))

    with (torch.no_grad()):
        with torch.autocast('cuda'):  # AMP for acceleration

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            if True:  # inverse the source video latent
                inverse_step = 25
                source_c = {}
                source_uc = {}
                edit_c = {}
                edit_uc = {}
                for key in c.keys():
                    source_c[key], edit_c[key] = c[key].chunk(2)
                    source_uc[key], edit_uc[key] = uc[key].chunk(2)

                additional_model_inputs["image_only_indicator"] = torch.zeros(2*1, num_frames).to(device)
                inversed_noise = model.sampler.my_inverse(denoiser, model, additional_model_inputs, latent, inverse_step=inverse_step, cond=source_c, uc=source_uc)
                additional_model_inputs["image_only_indicator"] = torch.zeros(2*bs, num_frames).to(device)
                appearanced_broken_noise = brokenV1_energy(inversed_noise, inverse_alpha)


                # appearanced_broken_noise = brokenV2_lowpass(inversed_noise, inverse_alpha)
                # appearanced_broken_noise = brokenV3_reduce(inversed_noise, inverse_alpha)
                # appearanced_broken_noise = brokenV4_subtemporalmean(inversed_noise, inverse_alpha)
                # appearanced_broken_noise = brokenV5_subspatialmean(inversed_noise, inverse_alpha)
                # ====================save inverted noise
                # from other_utils.imgs2video.imgs2video import imgs2video
                # imgs2video(inversed_noise[:, :3]/6+3, 'inversed_noise.mp4')
                # imgs2video(appearanced_broken_noise[:, :3], 'reduced_noise.mp4')
                # ========================================
                inversed_noise = torch.cat([inversed_noise, appearanced_broken_noise], dim=0)
                samples_z = model.sampler.my_denoising(denoiser, model, inversed_noise, depth_constraints, inverse_step=inverse_step, cond=c, uc=uc)



                # todo if perform alignment
                samples_reconstruct, samples_edit = samples_z.chunk(2)
                samples_reconstruct = (samples_reconstruct - samples_reconstruct.mean((1, 2, 3),  keepdims=True)) / samples_reconstruct.std((1, 2, 3), keepdims=True)
                samples_reconstruct = samples_reconstruct * latent.std((1, 2, 3), keepdims=True) + latent.mean((1, 2, 3), keepdims=True)

                samples_edit = (samples_edit - samples_edit.mean((1, 2, 3), keepdims=True)) / samples_edit.std((1, 2, 3), keepdims=True)
                samples_edit = samples_edit * edit_first_frame_latent.std() + edit_first_frame_latent.mean()

                mse = ((samples_reconstruct - latent) ** 2).mean()
                print(f'reconstruction error is {mse}.')

                return samples_reconstruct, samples_edit


def refine_video_clip(model,
                  latent,
                  input_images,
                  num_frames: Optional[int] = None,  # 21 for SV3D
                  noise_step: int=0,
                  num_steps: Optional[int] = None,
                  fps_id: int = 6,
                  motion_bucket_id: int = 127,
                  cond_aug: float = 0.02,
                  device: str = "cuda",
                  ):
    '''
    perform refine with num_steps diffusion forward and stochastic sampler
    args:
        model: SVD
        latent: [bf, c, h, w] the latent
        input_images: [1, c, h, w] the first RGB image a.k.a., first-frame condition
        num_frames: int
        noise_step: int range [0, 1000], demonstrate the added noise scale, 1000= all noise, 0= no noise
        num_steps: int
        device: device
    return:
        samples_refined: [bf, c, h, w] the shape of latent is same to input latent
    '''

    output_device = device
    device = model.device
    latent = latent.to('cpu').to(device)
    input_images = input_images.to('cpu').to(device)

    noise_step = int(noise_step/1000 * num_steps)

    # prepare data
    bs = input_images.shape[0]
    additional_model_inputs = {}
    additional_model_inputs["image_only_indicator"] = torch.zeros(2*bs, num_frames).to(device)  # 2 for (uc,c)
    additional_model_inputs["num_video_frames"] = num_frames
    assert input_images.shape[1] == 3
    shape = latent.shape
    c, uc = create_batch(model, input_images, motion_bucket_id, fps_id, cond_aug, num_frames, device)

    with (torch.no_grad()):
        with torch.autocast('cuda'):  # AMP for acceleration
            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )
            # without condition
            refined_x = model.sampler.my_refine(denoiser, latent, c, uc, noise_step=noise_step, num_steps=num_steps)

            return refined_x

