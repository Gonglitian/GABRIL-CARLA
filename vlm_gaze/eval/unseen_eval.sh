#!/bin/bash
# 使用示例: ./unseen_eval.sh [model] [traffic-manager-port] [port] [cuda-device]

# 示例: ./unseen_eval.sh pseudo_gmd 3000 6000 0 true

routes=(18305 1852 24224 3099 3184 3464 27529 26401 2215 25951)

# todo: change dir
cd /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/vlm_gaze/eval

# 检查是否提供了model参数
if [ $# -eq 0 ]; then
    model=pseudo_gmd
    echo "使用默认model: $model"
else
    model=$1
    echo "使用指定的model: $model"
fi

# 检查是否提供了traffic-manager-port参数
if [ $# -lt 2 ]; then
    traffic_manager_port=3000
    echo "使用默认traffic-manager-port: $traffic_manager_port"
else
    traffic_manager_port=$2
    echo "使用指定的traffic-manager-port: $traffic_manager_port"
fi

# 检查是否提供了port参数
if [ $# -lt 3 ]; then
    port=6000
    echo "使用默认port: $port"
else
    port=$3
    echo "使用指定的port: $port"
fi

# 检查是否提供了CUDA_VISIBLE_DEVICES参数
if [ $# -lt 4 ]; then
    cuda_device=0
    echo "使用默认CUDA_VISIBLE_DEVICES: $cuda_device"
else
    cuda_device=$4
    echo "使用指定的CUDA_VISIBLE_DEVICES: $cuda_device"
fi

# 检查是否提供了confounded参数（第5个参数，true/false）
if [ $# -lt 5 ]; then
    confounded=false
    echo "使用默认confounded: $confounded"
else
    confounded=$5
    echo "使用指定的confounded: $confounded"
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$cuda_device

# 生成随机seed函数
generate_random_seeds() {
    local seeds=()
    for i in {1..4}; do
        # 生成0-999之间的随机数
        seed=$((RANDOM % 1000))
        seeds+=($seed)
    done
    echo "${seeds[@]}"
}

echo "开始运行未见过的路由评估，总共 ${#routes[@]} 个路由"

for route_id in "${routes[@]}"; do
    echo "正在运行未见过的路由: $route_id"
    
    # 为当前路由生成4个随机seed
    random_seeds=($(generate_random_seeds))
    echo "使用的seeds: ${random_seeds[@]}"
    
    for seed in "${random_seeds[@]}"; do
        echo "  运行seed: $seed"
        if [ "$confounded" = true ]; then
            python env_manager.py --agent=BC --params_path=results/$model --seed=$seed --routes-id=$route_id --video_path=auto --traffic-manager-port=$traffic_manager_port --port=$port --confounded
        else
            python env_manager.py --agent=BC --params_path=results/$model --seed=$seed --routes-id=$route_id --video_path=auto --traffic-manager-port=$traffic_manager_port --port=$port
        fi
        echo "  seed $seed 运行完成"
    done
    
    echo "未见过的路由 $route_id 运行完成"
    echo "----------------------------------------"
done

echo "所有未见过的路由运行完成！" 