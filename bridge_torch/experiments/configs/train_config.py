from ml_collections import ConfigDict


def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        val_batch_size=256,
        num_steps=int(2e6),
        log_interval=100,
        eval_interval=5000,
        eval_batches=0,  # <=0 表示不限制
        save_interval=5000,
        save_dir="path/to/save/dir",
        data_path="path/to/data",
        resume_path=None,
        seed=42,
        device="cuda",  # cuda | cpu
        ddp=dict(
            enabled=False,
            find_unused_parameters=False,
        ),
        # 性能优化默认配置（用于 CLI 覆盖）
        amp=dict(
            enabled=False,
            dtype="bf16",  # bf16 | fp16
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        ),
        compile=dict(
            enabled=False,
            kwargs=dict(mode="default", dynamic=False, fullgraph=False),
        ),
        dataloader=dict(
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=True,
        ),
        profiler=dict(
            enabled=False,
            wait=1,
            warmup=1,
            active=5,
            repeat=1,
            record_shapes=True,
            with_stack=False,
        ),
        # 训练学习率调度策略
        # type: "constant" | "warmup_cosine"
        #   - constant: 学习率恒定为 learning_rate
        #   - warmup_cosine: 先 LinearLR 预热到 learning_rate，再 CosineAnnealingLR 衰减
        scheduler=dict(
            type="warmup_cosine",
        ),
        # WandB 设置，可由 YAML 覆盖
        wandb=dict(
            project="bridgedata_torch",
        ),
    )

    base_data_config = dict(
        shuffle_buffer_size=25000,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            random_brightness=(0.2,),
            random_contrast=(0.8, 1.2),
            random_saturation=(0.8, 1.2),
            random_hue=(0.1,),
            augment_order=(
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ),
        ),
    )

    # Only keep torch-implemented algos here: bc, gc_bc
    possible_structures = {
        "bc": ConfigDict(
            dict(
                agent="bc",
                agent_kwargs=dict(
                    network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                        # Saliency regularization (Reg variant) config
                        saliency=dict(
                            enabled=False,
                            weight=0.0,
                            beta=1.0,
                        ),
                    ),
                    use_proprio=False,
                    learning_rate=3e-4,
                    weight_decay=0.0,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    augment=True,
                    obs_horizon=1,
                    act_pred_horizon=1,
                    saliency_alpha=1.0,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    # default backbone arch to allow CLI overrides like
                    # --config.encoder_kwargs.arch resnet50
                    arch="resnet34",
                ),
                **base_real_config,
            )
        ),
        "gc_bc": ConfigDict(
            dict(
                agent="gc_bc",
                agent_kwargs=dict(
                    network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                        # Saliency regularization (Reg variant) config
                        saliency=dict(
                            enabled=False,
                            weight=0.0,
                            beta=1.0,
                        ),
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    weight_decay=0.0,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    augment=True,
                    obs_horizon=1,
                    act_pred_horizon=1,
                    saliency_alpha=1.0,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    arch="resnet34",
                ),
                **base_real_config,
            )
        ),
        "gc_ddpm_bc": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    # Saliency regularization (Reg variant) config
                    saliency=dict(
                        enabled=False,
                        weight=0.0,
                        beta=1.0,
                    ),
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                    target_update_rate=0.002,
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=1,
                    augment=True,
                    saliency_alpha=1.0,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    arch="resnet34",
                ),
                **base_real_config,
            )
        ),
    }

    if config_string not in possible_structures:
        raise ValueError(
            f"Unsupported algo for torch port: {config_string}. Use one of {list(possible_structures.keys())}"
        )

    return possible_structures[config_string]
