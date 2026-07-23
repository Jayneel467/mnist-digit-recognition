"""Configuration management for MNIST Digit Recognition."""

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model architecture
    INPUT_SHAPE: tuple = (28, 28)
    NUM_CLASSES: int = 10
    HIDDEN_UNITS_1: int = 128
    HIDDEN_UNITS_2: int = 128
    DROPOUT_RATE: float = 0.2

    # Training parameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.1
    RANDOM_SEED: int = 42

    # Optimization
    OPTIMIZER: str = "adam"
    LOSS_FUNCTION: str = "sparse_categorical_crossentropy"
    METRICS: list = None

    # Paths
    MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "models/digit_recognition.h5")
    CHECKPOINT_PATH: str = os.getenv("CHECKPOINT_PATH", "models/checkpoints")
    LOG_PATH: str = os.getenv("LOG_PATH", "logs")

    # Callbacks
    EARLY_STOPPING_PATIENCE: int = 3
    EARLY_STOPPING_MIN_DELTA: float = 0.001

    def __post_init__(self):
        """Post-initialization setup."""
        if self.METRICS is None:
            self.METRICS = ["accuracy"]


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/digit_recognition.h5")

    # Inference parameters
    CONFIDENCE_THRESHOLD: float = 0.5
    BATCH_SIZE: int = 32

    # Image preprocessing
    IMAGE_SIZE: tuple = (28, 28)
    NORMALIZE: bool = True

    # Logging
    LOG_PREDICTIONS: bool = True
    VERBOSE: bool = False


def get_training_config() -> TrainingConfig:
    """Get training configuration."""
    return TrainingConfig()


def get_inference_config() -> InferenceConfig:
    """Get inference configuration."""
    return InferenceConfig()
