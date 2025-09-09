# run torch version eval
python -m bridge_torch.experiments.eval \
  --save_dir /scr/litian/torch_runs/bridgedata_torch/gc_bc_lift_carrot_100_20250909_083848 \
  --goal_type gc \
  --im_size 256 \
  --video_save_path /scr/litian/torch_runs/videos \
  --ip localhost --port 5556 \
  --num_timesteps 120 --act_exec_horizon 1 --deterministic --show_image

# run training with nohup
nohup python bridge_torch/experiments/multi_train.py\
  --config bridge_torch/experiments/configs/multi_train.yaml > log.txt 2>&1 &

# upload eval results to google drive
rclone copy /scr/litian/torch_runs/bridgedata_torch_0909_eval litian_LIRA:Data/ --progress