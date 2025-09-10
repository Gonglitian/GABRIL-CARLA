# Refactored VLM-GABRIL Processing Pipeline

This directory contains the refactored version of the VLM-GABRIL processing pipeline with YAML-based configuration management. All hardcoded parameters have been extracted to configuration files for improved maintainability and flexibility.

## 📁 Directory Structure

```
refactor/
├── configs/                          # YAML configuration files
│   ├── pipeline_config.yaml         # Shared pipeline settings
│   ├── global_desc_config.yaml      # Global description generator settings
│   └── vlm_filter_config.yaml       # VLM filter settings
├── get_global_desc.py               # Refactored global description generator
├── pipeline_utils.py                # Refactored utility functions with config support
├── vlm_fliter.py                    # Original VLM filter (partially refactored)
├── vlm_filter.py         # Fully refactored VLM filter
├── test_config.py                   # Configuration testing suite
└── README.md                        # This file
```

## 🔧 Configuration Files

### `pipeline_config.yaml`
Shared configuration for all pipeline components:
- **API Settings**: SiliconFlow API configuration, model IDs
- **Processing Parameters**: Random seeds, image formats, thresholds
- **Routes & Seeds**: List of all route/seed pairs for batch processing
- **Action Processing**: Thresholds for CARLA action interpretation
- **Visualization**: Color palettes and display settings

### `global_desc_config.yaml`
Configuration for global description generation (`get_global_desc.py`):
- **Data Configuration**: Dataset paths, sampling parameters
- **Run Mode**: Single run vs. batch processing settings
- **VLM Configuration**: Model selection and prompt templates
- **Output Configuration**: Save paths and JSON formatting

### `vlm_filter_config.yaml`
Configuration for VLM-based object filtering (`vlm_fliter.py`):
- **Model Configuration**: Grounding DINO model and device settings
- **Data Configuration**: Dataset paths and processing parameters
- **Run Mode Configuration**: Single seed/route/all modes
- **Detection Settings**: Text prompts, thresholds, tracking parameters, and ability to use precomputed detections
- **VLM Filtering**: Enable/disable, detail levels, redetection settings
- **Output Configuration**: Save paths, file naming, visualization settings

## 🚀 Usage

### Running Global Description Generation

```bash
# Edit global_desc_config.yaml to set your parameters
python get_global_desc.py
```

Key configuration options:
```yaml
run_mode:
  mode: "single_run"  # or "batch"
  single_run:
    route: "route_2416"
    seed_name: "seed_200"

data:
  processing:
    k_frames: 20  # Number of frames to sample

vlm:
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
```

### Running VLM Filtering Pipeline

```bash
# Use the fully refactored version
python vlm_filter.py

# Or the partially refactored original
python vlm_fliter.py
```

Key configuration options:
```yaml
run_mode:
  mode: "single_route"  # "single_seed", "single_route", "all"
  single_route:
    route_id: 2416
  multithreading:
    enabled: true       # per-route threads when mode == all
    mode: "per_route"
    max_threads: 10

vlm_filtering:
  enabled: true
  detail_level: "low"

detection:
  box_threshold: 0.4
  text_prompt: "car. person. traffic light. traffic sign. bicycle. motorcycle."
  # Set to true to skip detection and read existing JSONs under
  # {existing_json.base_dir or output.base_output_dir}/{route}/{seed}/grounding_detections.json
  use_existing_json: true
  existing_json:
    base_dir: "/home/vla-reasoning/proj/vlm-gabril/Grounded-SAM-2/bench2drive_processor/refactor/results"
    filename: "grounding_detections.json"
```

### Configuration Testing

```bash
# Run the configuration test suite
python test_config.py
```

This verifies that all YAML files are valid and contain required sections.

## 📊 Refactoring Benefits

### ✅ Before Refactoring
- Hardcoded parameters scattered throughout code
- Difficult to modify settings without code changes
- No centralized configuration management
- Parameters duplicated across multiple files
- Hard to maintain consistency between components

### ✅ After Refactoring
- All parameters in centralized YAML configuration files
- Easy to modify settings without touching code
- Configuration validation and error handling
- Shared parameters defined once and reused
- Clear separation of code logic and configuration
- Support for environment variable overrides (API keys)
- Backward compatibility maintained where possible

## 🔑 Key Features

### Configuration Loading
```python
from pipeline_utils import load_pipeline_config, get_api_client

# Load shared configuration
config = load_pipeline_config()

# Get configured API client
client = get_api_client()

# Access configuration values
api_url = config["api"]["siliconflow"]["base_url"]
routes_seeds = config["routes_seeds"]["pairs"]
```

### Dynamic Parameter Support
- **API Keys**: Loaded from environment variables with fallbacks
- **Model Selection**: Configurable model IDs for different components
- **Thresholds**: All detection and tracking thresholds configurable
- **Paths**: All input/output paths configurable
- **Run Modes**: Flexible execution modes (single/batch/all)

### Configuration Validation
- YAML syntax validation on load
- Required section checking
- Type validation for critical parameters
- Helpful error messages for missing configurations

## 🛠 Extending the Configuration

To add new configuration parameters:

1. **Add to appropriate YAML file**:
```yaml
# In vlm_filter_config.yaml
my_new_feature:
  enabled: true
  threshold: 0.5
  options: ["option1", "option2"]
```

2. **Access in Python code**:
```python
config = load_vlm_filter_config()
if config["my_new_feature"]["enabled"]:
    threshold = config["my_new_feature"]["threshold"]
    # Use the configuration
```

3. **Update test suite**:
```python
# In test_config.py
assert "my_new_feature" in config, "Missing 'my_new_feature' section"
```

## 🔄 Migration Guide

### From Original to Refactored

1. **Replace hardcoded parameters**:
   ```python
   # Old way
   PARAMS = {"box_threshold": 0.4, "device": "cuda:7"}
   
   # New way
   config = load_vlm_filter_config()
   box_threshold = config["detection"]["box_threshold"]
   device = config["model"]["device"]
   ```

2. **Update function signatures**:
   ```python
   # Old way
   def process_data(params):
       device = params["device"]
   
   # New way
   def process_data(config):
       device = config["model"]["device"]
   ```

3. **Use configuration loading**:
   ```python
   # Add at top of script
   from pipeline_utils import load_pipeline_config
   config = load_pipeline_config()
   ```

## 🧪 Testing

The configuration system includes comprehensive tests:

- **YAML Syntax**: Validates all configuration files load correctly
- **Required Sections**: Ensures all necessary configuration sections exist
- **Integration**: Tests that shared values are consistent across configs
- **Value Types**: Validates that configuration values are correct types

Run tests with:
```bash
python test_config.py
```

## 📝 Configuration Reference

### Environment Variables
- `SILICONFLOW_API_KEY`: API key for VLM services (overrides config default)

### File Paths
All paths in configuration files can be:
- **Absolute**: `/full/path/to/directory`
- **Relative**: `relative/path` (relative to script location)

### Run Modes
- `single_seed`: Process one specific route/seed combination
- `single_route`: Process one route with all seeds (200-219)
- `batch`/`all`: Process all configured route/seed combinations

## 🐛 Troubleshooting

### Common Issues

1. **Configuration file not found**:
   ```
   FileNotFoundError: Configuration file not found: configs/pipeline_config.yaml
   ```
   - Ensure you're running from the correct directory
   - Check that config files exist in the `configs/` subdirectory

2. **Missing configuration section**:
   ```
   KeyError: 'api'
   ```
   - Check YAML syntax in configuration files
   - Ensure all required sections are present
   - Run `python test_config.py` to validate

3. **API key issues**:
   ```
   Authentication failed
   ```
   - Set `SILICONFLOW_API_KEY` environment variable
   - Or update the default key in `pipeline_config.yaml`

4. **Path not found**:
   ```
   FileNotFoundError: /data3/vla-reasoning/dataset/bench2drive220
   ```
   - Update data directory paths in configuration files
   - Ensure dataset is available at specified location

## 🎯 Future Improvements

- Add configuration schema validation using JSON Schema
- Implement configuration profiles for different environments
- Add command-line configuration overrides
- Create web-based configuration editor
- Add configuration versioning and migration tools
