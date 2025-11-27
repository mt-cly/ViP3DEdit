export CUDA_VISIBLE_DEVICES=0,1

python launch.py --config configs/vip3de.yaml --train  \
    trainer.max_steps=600 \
    trainer.val_check_interval=600 \
    system.prompt_processor.prompt="make it on fire." \
    data.source="./data/trex/" \
    data.max_view_num=25 \
    data.height=512 \
    data.width=768 \
    system.seed=18000 \
    system.guidance.guidance_scale=12.5 \
    system.guidance.condition_scale=1.3 \
    system.guidance.svd_height=512 \
    system.guidance.svd_width=768 \
    system.guidance.svd_min_scale=1.5 \
    system.guidance.svd_max_scale=2.5 \
    system.gs_source="./data/trex/gaussians/point_cloud/iteration_30000/point_cloud.ply" \
    system.source_prompt="A trex." \
    system.target_prompt="A trex on fire." \
    system.guidance.inverse_alpha=0.15 


