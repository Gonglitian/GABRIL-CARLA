import ml_collections
from experiments.configs.data_config import ACTION_PROPRIO_METADATA

def get_config(config_string="default"):
    assert config_string in task_list or config_string == "default", f"Task {config_string} not in {task_list}"
    task_list = ["open_microwave", "put_in_pot_lid", "remove_pot_lid"]
    selected_task = config_string
    return ml_collections.ConfigDict({
        # "include": [["open_microwave", "put_in_pot_lid", "remove_pot_lid"]],
        "include": [[selected_task]],
        "exclude": [],
        "sample_weights": None,
        "action_proprio_metadata": ACTION_PROPRIO_METADATA,
    })