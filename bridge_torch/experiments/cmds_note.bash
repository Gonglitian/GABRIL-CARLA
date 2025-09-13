# run torch version eval
python eval.py \
    --runs_root /data3/vla-reasoning/torch_runs/bridgedata_torch \
    --goal_type bc \
    --im_size 256 \
    --video_save_path /data3/vla-reasoning/torch_runs/videos \
    --ip localhost \
    --port 5556 \
    --num_timesteps 120 \
    --act_exec_horizon 1 \
    --deterministic \
    --show_image

# run numpy data convert 
python bridge_torch/data/bdv2_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.99 --im_size 256 --saliency

# DDP (4 GPUs) + Hydra（注意同时打开 ddp.enabled）
CUDA_VISIBLE_DEVICES=4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
    name=lift_carrot_100_proprio_w_saliency \
    algo.model.use_proprio=true \
    saliency.enabled=true \
    batch_size=1800 \
    num_steps=20000 \
    eval_interval=500 \
    save_interval=1000 \
    log_interval=10

python -m bridge_torch.experiments.train_hydra batch_size=2400 num_steps=5000 save_interval=1000 agent_kwargs.warmup_steps=500 ddp.enabled=false
# optional: 关闭 WandB / 打开混合精度与 compile
python -m bridge_torch.experiments.train_hydra \
  algo=gc_bc bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy save_dir=/path/to/runs \
  wandb.enabled=false \
  amp.enabled=true amp.dtype=bf16 \
  compile.enabled=true compile.kwargs.mode=default

# run training with nohup (Hydra, 带日志重定向)
nohup python -m bridge_torch.experiments.train_hydra \
  algo=bc bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy save_dir=/path/to/runs \
  batch_size=256 num_steps=30000 \
  saliency.enabled=true saliency.weight=0.2 > log.txt 2>&1 &

# upload eval results to google drive
rclone copy /scr/litian/torch_runs/bridgedata_torch_0909_eval litian_LIRA:Data/ --progress
