CUDA_VISIBLE_DEVICES=2,4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
  bridgedata=lift_carrot_mixed \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  saliency.weight=0 \
  saliency.alpha=0 \
  batch_size=2000 \
  num_steps=10000 \
  eval_interval=500 \
  save_interval=500 \
  log_interval=10

CUDA_VISIBLE_DEVICES=2,4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
  bridgedata=pull_pot_100 \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  saliency.weight=0 \
  saliency.alpha=0 \
  batch_size=2000 \
  num_steps=10000 \
  eval_interval=500 \
  save_interval=500 \
  log_interval=10

CUDA_VISIBLE_DEVICES=2,4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
  bridgedata=pull_pot_mixed \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  saliency.weight=0 \
  saliency.alpha=0 \
  batch_size=2000 \
  num_steps=10000 \
  eval_interval=500 \
  save_interval=500 \
  log_interval=10

CUDA_VISIBLE_DEVICES=2,4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
  bridgedata=put_carrot_in_pot_100 \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  saliency.weight=0 \
  saliency.alpha=0 \
  batch_size=2000 \
  num_steps=10000 \
  eval_interval=500 \
  save_interval=500 \
  log_interval=10
