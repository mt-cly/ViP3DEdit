from dataclasses import dataclass, field

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
from torchvision.utils import save_image
import torch
import threestudio
import os
from threestudio.systems.base import BaseLift3DSystem

from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.sam import LangSAMTextSegmentor

def depth2color(depth_map, colormap=cv2.COLORMAP_JET):
    # 归一化与伽马校正
    depth_normalized =depth_map # cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_gamma = np.uint8(np.power(depth_normalized / 255.0, 0.6) * 255)

    # 应用OpenCV内置色图
    colored = cv2.applyColorMap(depth_gamma, colormap)

    # 边缘增强
    # edges = cv2.Sobel(depth_map, cv2.CV_64F, 1, 1, ksize=3)
    # colored = cv2.addWeighted(colored, 0.7,
    #                           cv2.cvtColor(np.uint8(edges * 255), cv2.COLOR_GRAY2BGR), 0.3, 0)
    return colored


@threestudio.register("vip3de-system")
class DGE_SVD(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gs_source: str = None

        per_editing_step: int = -1
        edit_begin_step: int = 0
        edit_until_step: int = 4000

        densify_until_iter: int = 4000
        densify_from_iter: int = 0
        densification_interval: int = 1 # 100
        max_densify_percent: float = 0.01

        gs_lr_scaler: float = 1
        gs_final_lr_scaler: float = 1
        color_lr_scaler: float = 1
        opacity_lr_scaler: float = 1
        scaling_lr_scaler: float = 1
        rotation_lr_scaler: float = 1

        # lr
        mask_thres: float = 0.4
        max_grad: float = 1e-7
        min_opacity: float = 0.005

        seg_prompt: str = ""

        # cache
        cache_overwrite: bool = True
        cache_dir: str = ""


        # anchor
        anchor_weight_init: float = 0.1
        anchor_weight_init_g0: float = 1.0
        anchor_weight_multiplier: float = 2
        
        training_args: dict = field(default_factory=dict)

        use_masked_image: bool = False
        local_edit: bool = False

        # seed
        seed: int = 0

        # guidance 
        camera_update_per_step: int = 500
        added_noise_schedule: List[int] = field(default_factory=[999, 200, 200, 21])


        # source_prompt
        source_prompt: str = None
        target_prompt: str = None
        

    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=self.cfg.anchor_weight_init_g0,
            anchor_weight_init=self.cfg.anchor_weight_init,
            anchor_weight_multiplier=self.cfg.anchor_weight_multiplier,
        )
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())

        folder_name = f"edit_cache_{self.cfg.guidance_type.split('.')[0]}"

        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join(folder_name, self.cfg.cache_dir)
        else:
            # self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))
            self.cache_dir = os.path.join(folder_name,
                                      self.cfg.gs_source.split('/')[2] + '_' + self.cfg.prompt_processor.prompt.replace(' ', '_'))


    @torch.no_grad()
    def update_mask(self, save_name="mask") -> None:
        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
            masks = []
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_path_viz = os.path.join(
                    mask_cache_dir, "viz_{:0>4d}.png".format(id)
                )

                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                if self.cfg.seg_prompt == '':
                    mask = torch.ones_like(self.origin_frames[id][...,0]).to(get_device())
                else:
                    mask = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)[
                        0
                    ].to(get_device())

                mask_to_save = (
                        mask[0]
                        .cpu()
                        .detach()[..., None]
                        .repeat(1, 1, 3)
                        .numpy()
                        .clip(0.0, 1.0)
                        * 255.0
                ).astype(np.uint8)
                cv2.imwrite(cur_path, mask_to_save)

                masked_image = self.origin_frames[id].detach().clone()[0]
                masked_image[mask[0].bool()] *= 0.3
                masked_image_to_save = (
                        masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                masked_image_to_save = cv2.cvtColor(
                    masked_image_to_save, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(cur_path_viz, masked_image_to_save)
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)
                masks.append(mask)

            weights /= weights_cnt + 1e-7

            selected_mask = weights > self.cfg.mask_thres
            selected_mask = selected_mask[:, 0]
            torch.save(selected_mask, gs_mask_path)
        else:
            print("load cache")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_mask = cv2.imread(cur_path)
                cur_mask = torch.tensor(
                    cur_mask / 255, device="cuda", dtype=torch.float32
                )[..., 0][None]
            selected_mask = torch.load(gs_mask_path)
        self.masks = torch.stack(masks)
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

    def on_validation_epoch_end(self):

        pass

    def forward(self, batch: Dict[str, Any], renderbackground=None, local=False) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        semantics = []
        masks = []
        self.viewspace_point_list = []
        self.gaussian.localize = local
        for id, cam in enumerate(batch["camera"]):

            render_pkg = render(cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)

            semantic_map = render(
                cam,
                self.gaussian,
                self.pipe,
                renderbackground,
                override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = semantic_map > 0.8
            # todo no mask
            # semantic_map = torch.ones_like(semantic_map).bool()
            semantic_map_viz = image.detach().clone()
            semantic_map_viz = semantic_map_viz.permute(
                1, 2, 0
            )  # 3 512 512 to 512 512 3
            semantic_map_viz[semantic_map] = 0.40 * semantic_map_viz[
                semantic_map
            ] + 0.60 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
            semantic_map_viz = semantic_map_viz.permute(
                2, 0, 1
            )  # 512 512 3 to 3 512 512

            semantics.append(semantic_map_viz)
            masks.append(semantic_map)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        self.gaussian.localize = False  # reverse

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        semantics = torch.stack(semantics, dim=0)
        masks = torch.stack(masks, dim=0)

        render_pkg["semantic"] = semantics
        render_pkg["masks"] = masks
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def render_all_view(self, cache_name):
        # save orig_render with train_h and train_w
        cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(range(self.trainer.datamodule.train_dataset.total_view_num)):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    rendered = self(cur_batch)
                    out = rendered["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                    depth = rendered["depth_3dgs"]
                    cur_depth_path = os.path.join(cache_dir, "{:0>4d}_depth.png".format(id))
                    out_to_save = (
                            depth[0].clip(3,255).cpu().detach().numpy()**1.5
                    ).clip(0, 255).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    # out_to_save = depth2color(out_to_save)
                    cv2.imwrite(cur_depth_path, out_to_save)

                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]



        # save orig_render with orig h ,w
        cache_dir_origRes = os.path.join(self.cache_dir, cache_name, 'orig_res')
        os.makedirs(cache_dir_origRes, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(range(self.trainer.datamodule.val_dataset.total_view_num)):
                cur_path = os.path.join(cache_dir_origRes, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.val_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                    }
                    out = self(cur_batch)["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)

        # todo
        # clip similarity
        self.evaluate_clip_score()



    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.cfg.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(
                    self.viewspace_point_list[0]
                )
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + self.viewspace_point_list[idx].grad
                    )
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )
                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, self.visibility_filter
                )
                # Densification
                if (
                        self.true_global_step >= self.cfg.densify_from_iter
                        and self.true_global_step % self.cfg.densification_interval == 0
                ):  # 500 100
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        5,
                    )

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            # self.trainer.datamodule.train_dataset.scene.cameras[idx]
            self.trainer.datamodule.val_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)

        batch_trainres = batch.copy()
        batch_trainres["camera"] = [
            # self.trainer.datamodule.train_dataset.scene.cameras[idx]
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out_trainres = self(batch_trainres)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-val/{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"render_it{self.true_global_step}-val/{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

            # trainres
            self.save_image_grid(
                f"render_it{self.true_global_step}-val_trainres_frame{self.dataset.cfg.max_view_num}_alpha{self.cfg.guidance.inverse_alpha}/{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out_trainres["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out_trainres["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out_trainres
                    else []
                # )
                # + (
                #     [
                #         {
                #             "type": "rgb",
                #             "img": out_trainres["semantic"][idx].moveaxis(0, -1),
                #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                #         }
                #     ]
                #     if "semantic" in out_trainres
                #     else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

            # train resolution depth
            # self.save_image_grid(
            #     f"render_it{self.true_global_step}-val_trainres/{batch['index'][idx]}_depth.png",
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out_trainres["depth_3dgs"][idx][...,None]/20,
            #             "kwargs": {"data_format": "HWC"},
            #         },
            #     ],
            #     name=f"validation_step_render_{idx}",
            #     step=self.true_global_step,
            # )



    def test_step(self, batch, batch_idx):
        only_rgb = True  # TODO add depth test step
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        batch["camera"] = [
            self.trainer.datamodule.val_dataset.scene.cameras[batch["index"]]
        ]
        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=5,
            name="test",
            step=self.true_global_step,
        )
        save_list = []
        for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        if len(save_list) > 0:
            self.save_image_grid(
                f"edited_images.png",
                save_list,
                name="edited_images",
                step=self.true_global_step,
            )
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )

        save_path = self.get_save_path(f"last.ply")
        print("save_path", save_path)
        self.gaussian.save_ply(save_path)

        # clip similarity
        self.evaluate_clip_score(use_edit_cache=False)

    def evaluate_clip_score(self, use_edit_cache=False):
        import random
        from threestudio.utils.clip_metrics import ClipSimilarity
        n_cameras = len(self.trainer.datamodule.train_dataset.scene.cameras)
        # eval_indexes = random.sample(range(n_cameras), 10)
        eval_indexes = [i for i in range(n_cameras)]

        source_imgs = torch.cat([self.origin_frames[idx]for idx in eval_indexes])
        source_imgs = source_imgs.permute(0, 3, 1, 2).cpu()

        if use_edit_cache:
            target_imgs = [self.edit_frames[idx].cpu() for idx in eval_indexes]
            target_imgs = torch.cat(target_imgs, dim=0).permute(0, 3, 1, 2)
        else:
            target_imgs = []
            with torch.no_grad():
                for idx in eval_indexes:
                    batch = {}
                    batch["camera"] = [
                        self.trainer.datamodule.train_dataset.scene.cameras[idx]
                    ]
                    target_imgs.append(self(batch)["comp_rgb"].cpu())
            target_imgs = torch.cat(target_imgs, dim=0).permute(0, 3, 1, 2)

        clip = ClipSimilarity()
        sim_0, sim_1, sim_direction, sim_image, temp_sim_0, temp_sim_2 = clip(source_imgs, target_imgs, self.cfg.source_prompt, self.cfg.target_prompt)

        print(f'sim_0: {sim_0.mean()}')
        print(f'sim_1: {sim_1.mean()}')
        print(f'sim_direction: {sim_direction.mean()}')
        print(f'sim_img: {sim_image.mean()}')
        print(f'len:{len(sim_0)}')
        pass

    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")

        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        self.view_num = len(self.view_list)
        opt = OptimizationParams(self.parser, self.trainer.max_steps, self.cfg.gs_lr_scaler, self.cfg.gs_final_lr_scaler, self.cfg.color_lr_scaler,
                                 self.cfg.opacity_lr_scaler, self.cfg.scaling_lr_scaler, self.cfg.rotation_lr_scaler, )
        self.gaussian.load_ply(self.cfg.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent

        self.pipe = PipelineParams(self.parser)
        opt = OmegaConf.create(vars(opt))
        opt.update(self.cfg.training_args)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret

    def edit_all_view(self, original_render_name, cache_name, update_camera=False, global_step=0, sampling_gap=1):
        """
        对所有视角进行编辑的核心方法
        
        参数说明:
        - original_render_name (str): 原始渲染结果的缓存目录名，用于加载未编辑的原始图像
        - cache_name (str): 编辑结果的缓存目录名，编辑后的图像将保存在这里
        - update_camera (bool): 是否更新相机参数，默认False
        - global_step (int): 当前全局训练步数，用于控制编辑流程
        - sampling_gap (int): 采样间隔，控制视角采样的密度
        
        主要功能:
        1. 设置guidance的最大步数，根据训练进度动态调整噪声调度
        2. 收集所有训练视角的相机参数和渲染图像
        3. 对相机进行排序（可选）
        4. 加载对应的原始渲染图像作为参考
        5. 调用guidance模块对所有视角图像进行批量编辑
        6. 保存编辑后的图像到缓存目录
        7. 更新edit_frames字典，存储每个视角的编辑结果
        """

        # if self.true_global_step >= self.cfg.camera_update_per_step:
        #     self.guidance.use_normal_unet()
        
        # 初始化编辑相关的数据结构
        self.edited_cams = []
        self.edit_frames = {}
        
        # 设置缓存目录路径
        cache_dir = os.path.join(self.cache_dir, cache_name)  # 编辑结果保存目录
        original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)  # 原始渲染结果目录
        os.makedirs(cache_dir, exist_ok=True)

        # 准备数据容器
        cameras = []      # 存储相机参数
        images = []       # 存储当前渲染的图像
        depths = []       # 存储深度图
        original_frames = []  # 存储原始参考图像
        
        # 动态调整噪声调度：根据训练进度选择合适的最大噪声步数
        t_max_step = self.cfg.added_noise_schedule  # 噪声调度列表，如[999, 200, 200, 21]
        # 根据当前步数和相机更新频率确定使用哪个噪声级别
        self.guidance.max_step = t_max_step[min(len(t_max_step)-1, self.true_global_step//self.cfg.camera_update_per_step)]
        with torch.inference_mode():
            # 第一阶段：收集所有训练视角的相机参数
            for id in self.view_list:
                cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])
            
            # 对相机进行排序（目前使用简单的顺序排列，注释掉了复杂的空间排序）
            # todo no need sort
            # sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            sorted_cam_idx = [idx for idx in range(len(cameras))]
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]  # 排序后的视角ID列表
            cams_sorted = [cameras[idx] for idx in sorted_cam_idx]         # 排序后的相机列表

            # 第二阶段：逐个视角进行渲染和数据准备
            for id in view_sorted:
                # 设置各种路径
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))  # 当前编辑结果保存路径
                original_image_path = os.path.join(original_render_cache_dir, "{:0>4d}.png".format(id))  # 原始图像路径
                
                # 获取当前视角的相机参数
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                
                # 渲染当前视角
                out_pkg = self(cur_batch)
                out = out_pkg["comp_rgb"]    # 渲染的RGB图像
                depth = out_pkg["depth"]     # 深度图
                
                # 如果使用掩码图像，则应用掩码
                if self.cfg.use_masked_image:
                    out = out * out_pkg["masks"].unsqueeze(-1)
                
                images.append(out)
                depths.append(depth)
                
                # 加载对应的原始渲染图像作为参考
                assert os.path.exists(original_image_path)
                cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]
                original_frames.append(self.origin_frames[id])
            # 第三阶段：数据预处理，将列表转换为张量
            images = torch.cat(images, dim=0)           # 合并所有渲染图像 [N, H, W, C]
            depths = torch.cat(depths, dim=0)           # 合并所有深度图 [N, H, W, C]
            original_frames = torch.cat(original_frames, dim=0)  # 合并所有原始参考图像 [N, H, W, C]

            # 第四阶段：调用guidance模块进行图像编辑
            # 这是整个方法的核心，将所有视角的图像批量送入guidance进行编辑
            edited_images = self.guidance(
                images,             # 当前渲染的图像
                original_frames,    # 原始参考图像
                depths,             # 深度图
                self.masks,         # 分割掩码
                # self.prompt_processor(),
                self.prompt_processor.cfg.prompt,  # 编辑提示词
                cams = cams_sorted, # 排序后的相机参数
                seed=self.cfg.seed  # 随机种子
            )
            
            # 第五阶段：保存编辑结果
            print(f'save to {cache_dir}')
            for view_index_tmp in range(len(self.view_list)):
                # 更新edit_frames字典，存储每个视角的编辑结果
                self.edit_frames[view_sorted[view_index_tmp]] = edited_images['edit_images'][view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C
                
                # 将编辑后的图像保存到磁盘
                cv2.imwrite(os.path.join(cache_dir, "{:0>4d}.png".format(view_sorted[view_index_tmp])), 
                           cv2.cvtColor(edited_images['construct_edit_images'][view_index_tmp].cpu().numpy()*255, cv2.COLOR_BGR2RGB))
    

    def sort_the_cameras_idx(self, cams):
        foward_vectos = [cam.R[:, 2] for cam in cams]
        foward_vectos = np.array(foward_vectos)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
        distances = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), 0, 1)) for cam in cams]
        sorted_cams = [cam for _, cam in sorted(zip(distances, cams), key=lambda pair: pair[0])]
        reference_axis = np.cross(most_left_vecotr, sorted_cams[1].R[:, 2])
        distances_with_sign = [np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) if np.dot(reference_axis,  np.cross(most_left_vecotr, cam.R[:, 2])) >= 0 else 2 * np.pi - np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) for cam in cams]
        
        sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(cams))), key=lambda pair: pair[0])]

        return sorted_cam_idx

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        self.set_continuous_view(global_step=self.true_global_step, sampling_gap=self.cfg.guidance.sampling_gap)

        self.update_mask()

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0 or self.cfg.loss.use_sds:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            
    def set_continuous_view(self,global_step, sampling_gap):
        self.trainer.datamodule.train_dataset.update_cameras_contiguous(random_seed=global_step+1, sampling_gap=sampling_gap)
        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        sorted_train_view_list = sorted(self.view_list)
        selected_views = torch.linspace(
            0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
        )
        self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]

    def training_step(self, batch, batch_idx):
        if self.true_global_step == 0 and self.cfg.guidance_type == 'vip3de-guidance' and not self.cfg.loss.use_sds:
            # first editing
            self.edit_all_view(original_render_name='origin_render', cache_name=f"edited_views_frame{self.dataset.cfg.max_view_num}_alpha{self.cfg.guidance.inverse_alpha}", update_camera=True)


        self.gaussian.update_learning_rate(self.true_global_step)
        batch_index = batch["index"]

        if isinstance(batch_index, int):
            batch_index = [batch_index]
        if self.cfg.guidance_type == 'vip3de-guidance':
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames:
                    batch_index[img_index] = self.view_list[img_index]

        out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]
        mask = out["masks"].unsqueeze(-1)
        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                # if cur_index not in self.edit_frames:
                #     cur_index = self.view_list[0]
                if (cur_index not in self.edit_frames or (
                        self.cfg.per_editing_step > 0
                        and self.cfg.edit_begin_step
                        < self.global_step
                        < self.cfg.edit_until_step
                        and self.global_step % self.cfg.per_editing_step == 0
                )) and 'dge' not in str(self.cfg.guidance_type) and not self.cfg.loss.use_sds:
                    print(self.cfg.guidance_type)
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )
                 
                    self.edit_frames[cur_index] = result["edit_images"].detach().clone()

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)
            if self.cfg.use_masked_image:
                print("use masked image")
                guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images * mask, gt_images * mask),
                "loss_p": self.perceptual_loss(
                    (images * mask).permute(0, 3, 1, 2).contiguous(),
                    (gt_images * mask ).permute(0, 3, 1, 2).contiguous(),
                ).sum(),
                }
            else:
                guidance_out = {
                    "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                    "loss_p": self.perceptual_loss(
                        images.permute(0, 3, 1, 2).contiguous(),
                        gt_images.permute(0, 3, 1, 2).contiguous(),
                    ).sum(),
                }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )
        # sds loss
        if self.cfg.loss.use_sds:
            prompt_utils = self.prompt_processor()
            self.guidance.cfg.use_sds = True
            guidance_out = self.guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                prompt_utils)  
            loss += guidance_out["loss_sds"] * self.cfg.loss.lambda_sds 

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
    
        return {"loss": loss}
