"""Neural network model for MNIST digit recognition."""

import logging
from typing import Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
from src.config import TrainingConfig
from src.data_loader import MNISTDataLoader

logger = logging.getLogger(__name__)


class MNISTModel:
    """MNIST digit recognition model."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize MNIST model.

        Args:
            config: TrainingConfig object. If None, uses default configuration.
        """
        self.config = config or TrainingConfig()
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the neural network model."""
        logger.info("Building model...")
        try:
            self.model = keras.models.Sequential(
                [
                    keras.layers.Flatten(input_shape=self.config.INPUT_SHAPE),
                    keras.layers.Dense(
                        units=self.config.HIDDEN_UNITS_1,
                        activation="relu",
                    ),
                    keras.layers.Dropout(self.config.DROPOUT_RATE),
                    keras.layers.Dense(
                        units=self.config.HIDDEN_UNITS_2,
                        activation="relu",
                    ),
                    keras.layers.Dropout(self.config.DROPOUT_RATE),
                    keras.layers.Dense(
                        units=self.config.NUM_CLASSES,
                        activation="softmax",
                    ),
                ]
            )
            logger.info("Model built successfully")
        except Exception as e:
            logger.error(f"Error building model: {e}")\n            raise\n\n    def compile_model(self) -> None:\n        \"\"\"Compile the model.\"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not built. Call _build_model() first.\")\n\n        logger.info(\"Compiling model...\")\n        try:\n            self.model.compile(\n                optimizer=self.config.OPTIMIZER,\n                loss=self.config.LOSS_FUNCTION,\n                metrics=self.config.METRICS,\n            )\n            logger.info(\"Model compiled successfully\")\n        except Exception as e:\n            logger.error(f\"Error compiling model: {e}\")\n            raise\n\n    def train(\n        self,\n        x_train=None,\n        y_train=None,\n        verbose: int = 1,\n    ) -> keras.callbacks.History:\n        \"\"\"Train the model.\n\n        Args:\n            x_train: Training images. If None, loads MNIST dataset.\n            y_train: Training labels. If None, loads MNIST dataset.\n            verbose: Verbosity level (0, 1, or 2).\n\n        Returns:\n            Training history object.\n        \"\"\"\n        if self.model is None:\n            self._build_model()\n        self.compile_model()\n\n        # Load data if not provided\n        if x_train is None or y_train is None:\n            logger.info(\"Loading MNIST dataset...\")\n            loader = MNISTDataLoader()\n            x_train, y_train, _, _ = loader.get_processed_data()\n\n        logger.info(f\"Starting training for {self.config.EPOCHS} epochs...\")\n        try:\n            self.history = self.model.fit(\n                x_train,\n                y_train,\n                epochs=self.config.EPOCHS,\n                batch_size=self.config.BATCH_SIZE,\n                validation_split=self.config.VALIDATION_SPLIT,\n                verbose=verbose,\n            )\n            logger.info(\"Training completed successfully\")\n            return self.history\n        except Exception as e:\n            logger.error(f\"Error during training: {e}\")\n            raise\n\n    def evaluate(\n        self,\n        x_test=None,\n        y_test=None,\n        verbose: int = 1,\n    ) -> Dict[str, float]:\n        \"\"\"Evaluate the model on test data.\n\n        Args:\n            x_test: Test images. If None, loads MNIST test set.\n            y_test: Test labels. If None, loads MNIST test set.\n            verbose: Verbosity level.\n\n        Returns:\n            Dictionary with evaluation metrics.\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not built. Train the model first.\")\n\n        # Load data if not provided\n        if x_test is None or y_test is None:\n            logger.info(\"Loading MNIST test dataset...\")\n            from src.data_loader import load_mnist_data\n            _, _, x_test, y_test = load_mnist_data()\n\n        logger.info(\"Evaluating model on test data...\")\n        try:\n            results = self.model.evaluate(x_test, y_test, verbose=verbose)\n            metrics = dict(zip(self.model.metrics_names, results))\n            logger.info(\n                f\"Evaluation results - Loss: {metrics['loss']:.4f}, \"\n                f\"Accuracy: {metrics['accuracy']:.4f}\"\n            )\n            return metrics\n        except Exception as e:\n            logger.error(f\"Error during evaluation: {e}\")\n            raise\n\n    def predict(self, x: Any) -> Any:\n        \"\"\"Make predictions on data.\n\n        Args:\n            x: Input data.\n\n        Returns:\n            Predictions.\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not built. Train the model first.\")\n\n        try:\n            return self.model.predict(x)\n        except Exception as e:\n            logger.error(f\"Error during prediction: {e}\")\n            raise\n\n    def save(self, filepath: str) -> None:\n        \"\"\"Save the model.\n\n        Args:\n            filepath: Path to save the model.\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not built. Train the model first.\")\n\n        logger.info(f\"Saving model to {filepath}...\")\n        try:\n            self.model.save(filepath)\n            logger.info(\"Model saved successfully\")\n        except Exception as e:\n            logger.error(f\"Error saving model: {e}\")\n            raise\n\n    def load(self, filepath: str) -> None:\n        \"\"\"Load a saved model.\n\n        Args:\n            filepath: Path to the saved model.\n        \"\"\"\n        logger.info(f\"Loading model from {filepath}...\")\n        try:\n            self.model = keras.models.load_model(filepath)\n            logger.info(\"Model loaded successfully\")\n        except Exception as e:\n            logger.error(f\"Error loading model: {e}\")\n            raise\n\n    def get_summary(self) -> None:\n        \"\"\"Print model summary.\"\"\"\n        if self.model is None:\n            raise ValueError(\"Model not built.\")\n        self.model.summary()\n