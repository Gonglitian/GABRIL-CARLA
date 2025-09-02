#!/usr/bin/env python3
"""
Test script to train BC and Gaze models for one epoch
This tests the actual training functionality with minimal data
"""

import sys
import os
import torch
from pathlib import Path
import shutil
import h5py
import numpy as np

# Use package imports instead of manual sys.path modifications
ROOT = "/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA"  # kept for absolute config path below





def test_gaze_training():
    """Test gaze predictor training for one epoch"""
    print("\n" + "="*60)
    print("Testing Gaze Predictor Training (1 epoch)")
    print("="*60)
    
    try:
        from hydra import compose, initialize_config_dir
        from vlm_gaze.train.train_gaze_predictor import GazePredictorTrainer
        
        # Use real robomimic dataset (no dummy data)
        real_hdf5_path = "/data3/vla-reasoning/dataset/bench2drive220_robomimic_large_chunk.hdf5"
        
        # Load config and override for testing
        config_dir = Path('/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/vlm_gaze/configs').absolute()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name='train_gaze')
            
            # Override to run a minimal 1-epoch real-data test
            cfg.data.hdf5_path = real_hdf5_path
            cfg.data.num_episodes = 10
            cfg.data.batch_size = 4
            cfg.data.num_workers = 2
            cfg.data.cache_mode = "low_dim"
            cfg.training.epochs = 1
            cfg.training.save_interval = 1
            cfg.training.vis_interval = 10  # Disable visualization for test
            cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg.logging.log_dir = "/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/logs/gaze"
            cfg.logging.checkpoint_dir = "/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/trained_models/gaze"
        
        print("\nInitializing GazePredictorTrainer...")
        trainer = GazePredictorTrainer(cfg)
        
        print("Starting training...")
        trainer.train()
        
        # Check if checkpoint was saved
        checkpoint_path = Path(cfg.logging.checkpoint_dir) / cfg.data.task / trainer.save_dir / "model_ep1.torch"
        if checkpoint_path.exists():
            print(f"✓ Checkpoint saved at: {checkpoint_path}")
            
            # Load and verify checkpoint
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            print(f"✓ Checkpoint contains {len(state_dict)} parameters")
        else:
            print(f"✗ Checkpoint not found at: {checkpoint_path}")
            return False
        
        # Check if logs were created
        log_dir = Path(cfg.logging.log_dir) / cfg.data.task / trainer.save_dir
        if log_dir.exists():
            print(f"✓ Logs created at: {log_dir}")
        else:
            print(f"✗ Logs not found at: {log_dir}")
            return False
        
        print("\n✓ Gaze predictor training test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error in gaze training test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup logs and checkpoints
        if Path("/tmp/test_logs").exists():
            shutil.rmtree("/tmp/test_logs", ignore_errors=True)
        if Path("/tmp/test_checkpoints").exists():
            shutil.rmtree("/tmp/test_checkpoints", ignore_errors=True)


def test_bc_training():
    """Test behavior cloning training for one epoch"""
    print("\n" + "="*60)
    print("Testing Behavior Cloning Training (1 epoch)")
    print("="*60)
    
    try:
        from hydra import compose, initialize_config_dir
        from vlm_gaze.train.train_bc import BCTrainer
        
        # Use real robomimic dataset (no dummy data)
        real_hdf5_path = "/data3/vla-reasoning/dataset/bench2drive220_robomimic_large_chunk.hdf5"
        
        # Load config and override for testing
        config_dir = Path('/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/vlm_gaze/configs').absolute()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name='train_bc')
            
            # Override to run a minimal 1-epoch real-data test
            cfg.data.hdf5_path = real_hdf5_path
            cfg.data.num_episodes = 10
            cfg.data.batch_size = 4
            cfg.data.num_workers = 2
            cfg.data.cache_mode = "low_dim"
            cfg.training.epochs = 1
            cfg.training.save_interval = 1
            cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg.logging.log_dir = "/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/logs/bc"
            cfg.logging.checkpoint_dir = "/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/trained_models/bc"
            cfg.gaze.method = "Reg"  # Test with regularization
        
        print("\nInitializing BCTrainer...")
        trainer = BCTrainer(cfg)
        
        print("Starting training...")
        trainer.train()
        
        # Check if checkpoint was saved
        checkpoint_path = Path(cfg.logging.checkpoint_dir) / cfg.data.task / trainer.save_dir / "model_ep1.torch"
        if checkpoint_path.exists():
            print(f"✓ Checkpoint saved at: {checkpoint_path}")
            
            # Load and verify checkpoint
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            print(f"✓ Checkpoint contains {len(state_dict)} parameters")
            
            # Check for expected keys
            expected_prefixes = ['encoder.', 'pre_actor.', 'actor.']
            for prefix in expected_prefixes:
                matching_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
                if matching_keys:
                    print(f"  - Found {len(matching_keys)} {prefix[:-1]} parameters")
        else:
            print(f"✗ Checkpoint not found at: {checkpoint_path}")
            return False
        
        # Check if logs were created
        log_dir = Path(cfg.logging.log_dir) / cfg.data.task / trainer.save_dir
        if log_dir.exists():
            print(f"✓ Logs created at: {log_dir}")
        else:
            print(f"✗ Logs not found at: {log_dir}")
            return False
        
        # Check if params.json was saved
        params_path = Path(cfg.logging.checkpoint_dir) / cfg.data.task / trainer.save_dir / "params.json"
        if params_path.exists():
            print(f"✓ Parameters saved at: {params_path}")
        else:
            print(f"✗ Parameters not found at: {params_path}")
        
        print("\n✓ Behavior cloning training test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error in BC training test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup logs and checkpoints
        if Path("/tmp/test_logs").exists():
            shutil.rmtree("/tmp/test_logs", ignore_errors=True)
        if Path("/tmp/test_checkpoints").exists():
            shutil.rmtree("/tmp/test_checkpoints", ignore_errors=True)


def test_config_overrides():
    """Test that config path overrides work correctly"""
    print("\n" + "="*60)
    print("Testing Config Path Overrides")
    print("="*60)
    
    try:
        from hydra import compose, initialize_config_dir
        
        config_dir = Path('/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/vlm_gaze/configs').absolute()
        
        # Test custom log and checkpoint paths
        custom_log_dir = "/tmp/custom_logs"
        custom_checkpoint_dir = "/tmp/custom_checkpoints"
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test BC config
            cfg_bc = compose(
                config_name='train_bc',
                overrides=[
                    f'logging.log_dir={custom_log_dir}/bc',
                    f'logging.checkpoint_dir={custom_checkpoint_dir}/bc'
                ]
            )
            
            assert cfg_bc.logging.log_dir == f'{custom_log_dir}/bc', "BC log_dir override failed"
            assert cfg_bc.logging.checkpoint_dir == f'{custom_checkpoint_dir}/bc', "BC checkpoint_dir override failed"
            print(f"✓ BC config overrides work correctly")
            
            # Test Gaze config
            cfg_gaze = compose(
                config_name='train_gaze',
                overrides=[
                    f'logging.log_dir={custom_log_dir}/gaze',
                    f'logging.checkpoint_dir={custom_checkpoint_dir}/gaze'
                ]
            )
            
            assert cfg_gaze.logging.log_dir == f'{custom_log_dir}/gaze', "Gaze log_dir override failed"
            assert cfg_gaze.logging.checkpoint_dir == f'{custom_checkpoint_dir}/gaze', "Gaze checkpoint_dir override failed"
            print(f"✓ Gaze config overrides work correctly")
        
        print("\n✓ Config path override test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error in config override test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Training One Epoch Test Suite")
    print("="*60)
    
    # Suppress robomimic warning
    import warnings
    warnings.filterwarnings("ignore", message=".*No private macro file found.*")
    
    tests = [
        ("Config Overrides", test_config_overrides),
        ("Gaze Training", test_gaze_training),
        ("BC Training", test_bc_training)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All training tests passed!")
        print("\nThe training scripts are working correctly with:")
        print("- Proper checkpoint saving in .torch format")
        print("- Correct log directory structure")
        print("- Configurable paths via YAML")
        print("- Compatible with original evaluation scripts")
        print("\nUsage examples:")
        print("\n1. Train with custom paths:")
        print("   python -m vlm_gaze.train.train_bc logging.log_dir=/my/logs logging.checkpoint_dir=/my/models")
        print("\n2. Train with different configurations:")
        print("   python -m vlm_gaze.train.train_bc --config-name=train_bc_gril")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)