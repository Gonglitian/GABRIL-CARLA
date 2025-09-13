# run torch version eval
python -m bridge_torch.experiments.eval \
  --save_dir /scr/litian/torch_runs/bridgedata_torch/gc_bc_lift_carrot_100_20250909_083848 \
  --goal_type gc \
  --im_size 256 \
  --video_save_path /scr/litian/torch_runs/videos \
  --ip localhost --port 5556 \
  --num_timesteps 120 --act_exec_horizon 1 --deterministic --show_image

# run numpy data convert 
python bridge_torch/data/bdv2_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.99 --im_size 256 --saliency

# single run (Hydra, BC)
python -m bridge_torch.experiments.train_hydra \
  algo=bc bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy \
  save_dir=/path/to/runs \
  batch_size=256 num_steps=30000 \
  encoder_kwargs.arch=resnet34 \
  dataset_kwargs.obs_horizon=1 dataset_kwargs.augment=true

# single run with saliency (统一顶层入口)
python -m bridge_torch.experiments.train_hydra \
  algo=bc bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy save_dir=/path/to/runs \
  batch_size=256 num_steps=30000 \
  saliency.enabled=true saliency.weight=0.2 saliency.beta=1.0 saliency.alpha=0.7

# hydra multirun sweep (多个算法/任务/种子与 saliency 权重)
python -m bridge_torch.experiments.train_hydra -m \
  algo=bc,gc_bc \
  bridgedata=lift_carrot_100,pull_pot_100 \
  seed=1,2,3 \
  saliency.enabled=true saliency.weight=0.1,0.2 \
  data_path=/path/to/bdv2_numpy save_dir=/path/to/runs

# DDP (4 GPUs) + Hydra（注意同时打开 ddp.enabled）
CUDA_VISIBLE_DEVICES=4,5,7 torchrun --nproc_per_node=4 -m bridge_torch.experiments.train_hydra \
  algo=gc_ddpm_bc bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy save_dir=/path/to/runs \
  batch_size=128 ddp.enabled=true

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
