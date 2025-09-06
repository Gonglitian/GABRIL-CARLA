#!/usr/bin/env python3
"""
BC 策略评估示例脚本

使用方法：
python eval_bc_example.py \
    --checkpoint_config_path /path/to/checkpoint/config.json \
    --checkpoint_path /path/to/checkpoint \
    --goal_type bc \
    --im_size 256
"""

import sys
import os

# 检查是否存在原始 eval.py
eval_script = os.path.join(os.path.dirname(__file__), 'experiments', 'eval.py')
if not os.path.exists(eval_script):
    print(f"错误：找不到 eval.py 脚本：{eval_script}")
    sys.exit(1)

print("BC 策略评估说明：")
print("="*50)
print("1. BC (Behavior Cloning) 策略不需要目标条件")
print("2. 使用 goal_type=bc 来评估 BC 策略")
print("3. 完整的评估命令示例：")
print()
print("python experiments/eval.py \\")
print("    --checkpoint_config_path /path/to/checkpoint/config.json \\")
print("    --checkpoint_path /path/to/checkpoint \\")
print("    --goal_type bc \\")
print("    --im_size 256 \\")
print("    --ip YOUR_ROBOT_IP \\")
print("    --port YOUR_ROBOT_PORT")
print()
print("注意事项：")
print("- BC 策略在评估时会使用虚拟目标（空图像）")
print("- 策略完全基于当前观察进行动作预测")
print("- 不需要提供任何目标图像或语言指令")
print()
print("训练好的 BC 模型通常保存在 $SAVE_DIR 目录下")
print("配置文件和检查点会在训练完成后自动生成")
