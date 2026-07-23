#!/usr/bin/env python
"""Training script for MNIST Digit Recognition model."""

import logging
import argparse
from pathlib import Path
from src.model import MNISTModel
from src.config import TrainingConfig, get_training_config
from src.utils import setup_logging, create_directories

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main(args):
    """Main training function.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("MNIST Digit Recognition - Training Pipeline")
    logger.info("=" * 80)
    
    # Create necessary directories
    create_directories([
        "models",
        "logs",
        "data",
    ])
    
    # Get configuration
    config = get_training_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    
    logger.info(f"Training Configuration:")
    logger.info(f"  - Epochs: {config.EPOCHS}")
    logger.info(f"  - Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  - Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  - Model Save Path: {config.MODEL_SAVE_PATH}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MNISTModel(config=config)
    model.get_summary()
    
    # Train model
    logger.info("Starting training...")
    try:
        history = model.train()
        logger.info("Training completed successfully!")
        
        # Save model
        logger.info(f"Saving model to {config.MODEL_SAVE_PATH}...")
        model.save(config.MODEL_SAVE_PATH)
        logger.info("Model saved successfully!")
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        metrics = model.evaluate()
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Loss: {metrics['loss']:.4f}")
        
        logger.info("=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MNIST Digit Recognition model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimizer",
    )
    
    args = parser.parse_args()
    main(args)
