"""Data loading and preprocessing for MNIST dataset."""

import logging
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from src.config import TrainingConfig

logger = logging.getLogger(__name__)


class MNISTDataLoader:
    """Load and preprocess MNIST dataset."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize data loader.

        Args:
            config: TrainingConfig object. If None, uses default configuration.
        """
        self.config = config or TrainingConfig()
        self.mnist = tf.keras.datasets.mnist
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST dataset.

        Returns:
            Tuple containing (x_train, y_train, x_test, y_test)
        """
        logger.info("Loading MNIST dataset...")
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (
                self.mnist.load_data()\n            )\n            logger.info(\n                f\"Dataset loaded successfully. \"\n                f\"Training samples: {len(self.x_train)}, \"\n                f\"Test samples: {len(self.x_test)}\"\n            )\n            return self.x_train, self.y_train, self.x_test, self.y_test\n        except Exception as e:\n            logger.error(f\"Error loading MNIST dataset: {e}\")\n            raise\n\n    def preprocess_data(\n        self,\n        x_train: Optional[np.ndarray] = None,\n        x_test: Optional[np.ndarray] = None,\n    ) -> Tuple[np.ndarray, np.ndarray]:\n        \"\"\"Preprocess images by normalizing pixel values.\n\n        Args:\n            x_train: Training images. If None, uses loaded data.\n            x_test: Test images. If None, uses loaded data.\n\n        Returns:\n            Tuple containing (x_train_processed, x_test_processed)\n        \"\"\"\n        if x_train is None:\n            x_train = self.x_train\n        if x_test is None:\n            x_test = self.x_test\n\n        if x_train is None or x_test is None:\n            raise ValueError(\"Data not loaded. Call load_data() first.\")\n\n        logger.info(\"Preprocessing data...\")\n        try:\n            x_train = tf.keras.utils.normalize(x_train, axis=1).numpy()\n            x_test = tf.keras.utils.normalize(x_test, axis=1).numpy()\n            logger.info(\"Data normalized successfully\")\n            return x_train, x_test\n        except Exception as e:\n            logger.error(f\"Error preprocessing data: {e}\")\n            raise\n\n    def get_processed_data(\n        self,\n    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n        \"\"\"Load and preprocess MNIST dataset in one call.\n\n        Returns:\n            Tuple containing (x_train, y_train, x_test, y_test) preprocessed\n        \"\"\"\n        self.load_data()\n        x_train, x_test = self.preprocess_data()\n        return x_train, self.y_train, x_test, self.y_test\n\n\ndef load_mnist_data(\n    normalize: bool = True,\n) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n    \"\"\"Convenience function to load and preprocess MNIST data.\n\n    Args:\n        normalize: Whether to normalize the data.\n\n    Returns:\n        Tuple containing (x_train, y_train, x_test, y_test)\n    \"\"\"\n    loader = MNISTDataLoader()\n    return loader.get_processed_data()\n