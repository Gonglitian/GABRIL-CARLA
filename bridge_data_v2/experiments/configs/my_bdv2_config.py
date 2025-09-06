import ml_collections
from experiments.configs.data_config import ACTION_PROPRIO_METADATA

def get_config(config_string="default"):
    assert config_string in task_list or config_string == "default", f"Task {config_string} not in {task_list}"
    task_list = ["open_microwave", "put_in_pot_lid", "remove_pot_lid"]
    selected_task = config_string
    return ml_collections.ConfigDict({
        # 一组任务列表（可只写你想训练的任务名）
        # 也可写通配符："?*" 表示所有一级子目录
        # "include": [["open_microwave", "put_in_pot_lid", "remove_pot_lid"]],
        "include": [[selected_task]],
        "exclude": [],
        "sample_weights": None,  # 多任务时可自定义采样权重
        "action_proprio_metadata": ACTION_PROPRIO_METADATA,
    })