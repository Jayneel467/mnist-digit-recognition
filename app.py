#!/usr/bin/env python
"""Flask API server for MNIST digit recognition inference."""

import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
from src.inference import DigitPredictor
from src.config import get_api_config, get_inference_config
from src.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
api_config = get_api_config()
inference_config = get_inference_config()

# Initialize predictor
try:
    predictor = DigitPredictor(
        model_path=api_config.MODEL_PATH,
        config=inference_config,
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint.
    
    Returns:
        JSON with health status
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None,
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """Predict digit from image.
    
    Expected JSON:
    {
        "image_path": "path/to/image.png"  # Optional
    }
    
    Or multipart form with 'file' field.
    
    Returns:
        JSON with prediction results
    """
    try:
        if predictor is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Handle file upload
        if "file" in request.files:
            file = request.files["file"]
            image_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                return jsonify({"error": "Invalid image format"}), 400
        
        # Handle image path
        elif "image_path" in request.json:
            image_path = request.json["image_path"]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                return jsonify({"error": f"Could not load image from {image_path}"}), 400
        
        else:
            return jsonify({"error": "Must provide 'file' or 'image_path'"}), 400
        
        # Make prediction
        result = predictor.predict(image)
        
        return jsonify({
            "success": True,
            "digit": result["digit"],
            "confidence": result["confidence"],
            "all_predictions": result["all_predictions"],
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch() -> Dict[str, Any]:
    """Predict digits from multiple images.
    
    Expected JSON:
    {
        "images": ["path1.png", "path2.png", ...]
    }
    
    Returns:
        JSON with batch prediction results
    """
    try:
        if predictor is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        image_paths = data.get("images", [])
        
        if not image_paths:
            return jsonify({"error": "Must provide 'images' list"}), 400
        
        results = []
        for image_path in image_paths:
            try:
                result = predictor.predict_from_file(image_path)
                results.append({
                    "image": image_path,
                    "success": True,
                    **result,
                })
            except Exception as e:
                results.append({
                    "image": image_path,
                    "success": False,
                    "error": str(e),
                })
        
        return jsonify({
            "success": True,
            "results": results,
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route("/api/model_info", methods=["GET"])
def model_info() -> Dict[str, Any]:
    """Get model information.
    
    Returns:
        JSON with model details
    """
    try:
        if predictor is None or predictor.model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        model = predictor.model
        
        return jsonify({
            "success": True,
            "model_type": "Sequential",
            "num_layers": len(model.layers),
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "total_params": int(model.count_params()),
            "trainable_params": sum(
                int(np.prod(w.shape)) 
                for w in model.trainable_weights
            ),
        }), 200
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.errorhandler(404)
def not_found(error) -> Dict[str, Any]:
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
    }), 404


@app.errorhandler(500)
def internal_error(error) -> Dict[str, Any]:
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error",
    }), 500


if __name__ == "__main__":
    logger.info(f"Starting API server on {api_config.HOST}:{api_config.PORT}")
    app.run(
        host=api_config.HOST,
        port=api_config.PORT,
        debug=api_config.DEBUG,
    )
