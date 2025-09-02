#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
from pathlib import Path

def calculate_route_average(method_name, seed=None, route_type="seen"):
    """
    计算指定方法在指定route类型下的score_composed平均值和方差
    
    Args:
        method_name
        seed: seed值，默认为None表示使用所有seed
        route_type: route类型，"seen"或"unseen"
    """
    # route列表
    if route_type == "seen":
        routes = [2416, 3100, 3472, 24211, 24258, 24759, 25857, 25863, 26408, 27494]
    elif route_type == "unseen":
        routes = [18305, 1852,  24224, 3099, 3184, 3464, 27529, 26401, 2215,  25951]
    else:
        raise ValueError("route_type必须是'seen'或'unseen'")
    
    # 存储所有score_composed值
    score_composeds = []
    successful_evaluations = []  # 存储(route_id, seed)对
    failed_evaluations = []
    
    if seed is None:
        print(f"正在计算 {method_name} 方法在所有seed下的{route_type} route平均值...")
    else:
        print(f"正在计算 {method_name} 方法在 seed_{seed} 下的{route_type} route平均值...")
    print(f"{route_type.capitalize()} routes数量: {len(routes)}")
    print("-" * 60)
    
    # 遍历所有route
    for route_id in routes:
        route_path = Path(f"{method_name}/route_{route_id}")
        
        if not route_path.exists():
            if route_type == "seen":
                print(f"警告: Route目录不存在 {route_path}")
            failed_evaluations.append((route_id, None, "Route目录不存在"))
            continue
        
        # 如果指定了seed，只处理该seed
        if seed is not None:
            seed_dirs = [f"seed_{seed}"]
        else:
            # 获取所有seed目录
            seed_dirs = [d.name for d in route_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
            
        if not seed_dirs:
            if route_type == "seen":
                print(f"警告: Route {route_id} 下没有找到seed目录")
            failed_evaluations.append((route_id, None, "没有seed目录"))
            continue
            
        # 遍历所有seed目录
        for seed_dir in seed_dirs:
            current_seed = seed_dir.split("_")[1] if "_" in seed_dir else seed_dir
            stats_path = route_path / seed_dir / "stats.json"
            
            if not stats_path.exists():
                if route_type == "seen":
                    print(f"警告: 文件不存在 {stats_path}")
                failed_evaluations.append((route_id, current_seed, "stats.json不存在"))
                continue
                
            try:
                # 读取并解析JSON文件
                with open(stats_path, 'r') as f:
                    data = json.load(f)
                
                # 提取score_composed值
                score_composed = data["_checkpoint"]["global_record"]["scores_mean"]["score_composed"]
                score_composeds.append(score_composed)
                successful_evaluations.append((route_id, current_seed))
                
                if route_type == "seen":
                    print(f"Route {route_id} seed_{current_seed}: score_composed = {score_composed}")
                
            except (json.JSONDecodeError, KeyError) as e:
                if route_type == "seen":
                    print(f"错误: 解析文件 {stats_path} 时出错: {e}")
                failed_evaluations.append((route_id, current_seed, f"JSON解析错误: {e}"))
                continue
            except Exception as e:
                if route_type == "seen":
                    print(f"错误: 处理文件 {stats_path} 时出错: {e}")
                failed_evaluations.append((route_id, current_seed, f"处理错误: {e}"))
                continue
    
    # 计算统计信息
    if score_composeds:
        # 转换为numpy数组便于计算
        scores_array = np.array(score_composeds)
        
        # 计算基本统计量
        mean_score = np.mean(scores_array)
        variance = np.var(scores_array, ddof=1)  # 样本方差
        std_dev = np.std(scores_array, ddof=1)   # 样本标准差
        std_error = std_dev / np.sqrt(len(scores_array))  # 标准误差
        
        print("-" * 60)
        print(f"结果统计:")
        print(f"成功处理的评估数量: {len(score_composeds)}")
        print(f"评估来自 {len(set(eval[0] for eval in successful_evaluations))} 个routes")
        print(f"总共 {len(routes)} 个routes")
        if route_type == "seen":
            print(f"所有score_composed值: {score_composeds}")
        print(f"平均值: {mean_score:.2f}")
        print(f"方差: {variance:.2f}")
        print(f"标准差: {std_dev:.2f}")
        print(f"标准误差: {std_error:.2f}")
        print(f"误差范围 (95%置信区间): [{mean_score - 1.96*std_error:.2f}, {mean_score + 1.96*std_error:.2f}]")
        print(f"误差范围 (68%置信区间): [{mean_score - std_error:.2f}, {mean_score + std_error:.2f}]")
        
        # 按route分组统计
        route_stats = {}
        for route_id, seed in successful_evaluations:
            if route_id not in route_stats:
                route_stats[route_id] = []
            # 找到对应的分数
            idx = successful_evaluations.index((route_id, seed))
            route_stats[route_id].append((seed, score_composeds[idx]))
        
        # 输出成功和失败的评估信息
        print("-" * 60)
        print(f"成功评估的routes和seeds ({len(successful_evaluations)}个评估):")
        if route_type == "seen":
            for route_id in sorted(route_stats.keys()):
                seeds_scores = route_stats[route_id]
                seeds_str = ', '.join([f"seed_{s}({sc:.1f})" for s, sc in seeds_scores])
                print(f"  Route {route_id}: {seeds_str}")
        else:
            # 对于unseen routes，只显示汇总信息
            for route_id in sorted(list(route_stats.keys())[:5]):
                seeds_scores = route_stats[route_id]
                print(f"  Route {route_id}: {len(seeds_scores)} seeds")
            if len(route_stats) > 5:
                print(f"  ... 还有 {len(route_stats) - 5} 个routes")
        
        if failed_evaluations:
            print(f"\n未成功评估的cases ({len(failed_evaluations)}个):")
            if route_type == "seen":
                for route_id, seed, reason in failed_evaluations:
                    seed_str = f"seed_{seed}" if seed else "未知seed"
                    print(f"  Route {route_id} {seed_str}: {reason}")
            else:
                # 对于unseen routes，只显示前10个失败的
                for route_id, seed, reason in failed_evaluations[:10]:
                    seed_str = f"seed_{seed}" if seed else "未知seed"
                    print(f"  Route {route_id} {seed_str}: {reason}")
                if len(failed_evaluations) > 10:
                    print(f"  ... 还有 {len(failed_evaluations) - 10} 个失败的评估")
        else:
            print(f"\n所有评估都成功了！")
        
        return {
            'mean': mean_score,
            'variance': variance,
            'std_dev': std_dev,
            'std_error': std_error,
            'count': len(score_composeds),
            'total_routes': len(routes),
            'successful_routes': len(route_stats),
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'route_stats': route_stats
        }
    else:
        print("错误: 没有成功处理任何评估")
        print(f"所有评估都失败了:")
        if route_type == "seen":
            for route_id, seed, reason in failed_evaluations:
                seed_str = f"seed_{seed}" if seed else "未知seed"
                print(f"  Route {route_id} {seed_str}: {reason}")
        else:
            for route_id, seed, reason in failed_evaluations[:10]:
                seed_str = f"seed_{seed}" if seed else "未知seed"
                print(f"  Route {route_id} {seed_str}: {reason}")
            if len(failed_evaluations) > 10:
                print(f"  ... 还有 {len(failed_evaluations) - 10} 个失败的评估")
        return None

def main():
    parser = argparse.ArgumentParser(description='计算route的score_composed平均值和方差')
    parser.add_argument('--method', '-m', default='pseudo_gmd', 
                       help='方法名称 (默认: pseudo_gmd)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='seed值 (默认: None表示使用所有seed)')
    parser.add_argument('--route-type', '-r', choices=['seen', 'unseen'], default='seen',
                       help='route类型: seen或unseen (默认: seen)')
    
    args = parser.parse_args()
    
    # 计算统计信息
    result = calculate_route_average(args.method, args.seed, args.route_type)
    
    if result:
        if args.seed is None:
            print(f"\n总结: {args.method} 在 {args.route_type} routes上所有seed的平均score_composed为 {result['mean']:.2f} ± {result['std_error']:.2f}")
            print(f"基于 {result['count']} 个评估 (来自 {result['successful_routes']} 个routes)")
        else:
            print(f"\n总结: {args.method} 在 {args.route_type} routes seed_{args.seed}上的平均score_composed为 {result['mean']:.2f} ± {result['std_error']:.2f}")

if __name__ == "__main__":
    main() 