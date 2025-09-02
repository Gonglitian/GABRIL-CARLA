"""
Training scripts for VLM-GAZE
"""

from .train_gaze_predictor import GazePredictorTrainer
from .train_bc import BCTrainer

__all__ = [
    'GazePredictorTrainer',
    'BCTrainer'
]