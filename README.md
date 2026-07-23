# mnist-digit-recognition

Production-ready MNIST digit recognition using TensorFlow/Keras with OpenCV integration

## Overview

This repository provides a complete, production-ready pipeline for training and deploying a neural network model for handwritten digit recognition on the MNIST dataset. It includes:

- **Model Training Pipeline**: Train a custom neural network with configurable hyperparameters
- **REST API Server**: Flask API with inference endpoints for single and batch predictions
- **Docker Support**: Multi-stage Dockerfile and docker-compose for containerized deployment
- **Image Preprocessing**: OpenCV integration for robust image handling and normalization
- **Logging & Configuration**: Environment-based config management with structured logging
- **Testing Ready**: Pytest setup with code quality tools (black, flake8, mypy, pylint)

## Features

- ✅ **TensorFlow/Keras** neural network with dropout regularization
- ✅ **OpenCV** image preprocessing (resize, normalization, grayscale conversion)
- ✅ **Flask REST API** with CORS support for cross-origin requests
- ✅ **Batch Prediction** endpoint for processing multiple images
- ✅ **Health Checks** and model info endpoints
- ✅ **Production Server** using Gunicorn with configurable workers
- ✅ **Docker Multi-Stage Build** for optimized container images
- ✅ **Comprehensive Logging** with structured JSON output
- ✅ **Code Quality Tools** (black, flake8, mypy, pylint, pytest)
- ✅ **Environment Configuration** with .env support

## Architecture

### Model Architecture

```
Input (28x28) 
    ↓
Flatten (784)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (10, Softmax)
    ↓
Output (digit 0-9 probabilities)
```

### Project Structure

```
.
├── app.py                    # Flask API server
├── train.py                  # Training entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Docker Compose configuration
├── .env.example             # Environment template
├── LICENSE                  # MIT License
│
└── src/
    ├── __init__.py
    ├── config.py            # Configuration classes (Training, Inference)
    ├── model.py             # MNISTModel class and training logic
    ├── data_loader.py       # MNISTDataLoader for dataset handling
    ├── inference.py         # DigitPredictor for making predictions
    └── utils.py             # Utility functions (logging, directories)
```

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jayneel467/mnist-digit-recognition.git
   cd mnist-digit-recognition
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration if needed
   ```

### Docker Setup

```bash
docker-compose up --build
```

## Quick Start

### Training

Train the model with default configuration:

```bash
python train.py
```

Train with custom hyperparameters:

```bash
python train.py --epochs 20 --batch-size 64 --learning-rate 0.0005
```

**Available arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Optimizer learning rate (default: 0.001)

**Output:**
- Trained model: `models/digit_recognition.h5`
- Training logs: `logs/` directory
- Checkpoints: `models/checkpoints/` (if enabled)

### Running the API Server

**Development mode:**
```bash
python app.py
```

**Production mode with Gunicorn:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Single Prediction
**Option A: File Upload**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@digit.png"
```

**Option B: Image Path**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path": "path/to/image.png"}'
```

Response:
```json
{
  "success": true,
  "digit": 7,
  "confidence": 0.9823,
  "all_predictions": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.0001,
    "3": 0.0001,
    "4": 0.0001,
    "5": 0.0001,
    "6": 0.0001,
    "7": 0.9823,
    "8": 0.0062,
    "9": 0.0007
  }
}
```

#### 3. Batch Prediction
```bash
curl -X POST http://localhost:5000/api/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["path/to/digit1.png", "path/to/digit2.png"]
  }'
```

Response:
```json
{
  "success": true,
  "results": [
    {
      "image": "path/to/digit1.png",
      "success": true,
      "digit": 3,
      "confidence": 0.9752,
      "all_predictions": {...}
    },
    {
      "image": "path/to/digit2.png",
      "success": true,
      "digit": 8,
      "confidence": 0.9641,
      "all_predictions": {...}
    }
  ]
}
```

#### 4. Model Info
```bash
curl http://localhost:5000/api/model_info
```

Response:
```json
{
  "success": true,
  "model_type": "Sequential",
  "num_layers": 6,
  "input_shape": [null, 28, 28],
  "output_shape": [null, 10],
  "total_params": 109930,
  "trainable_params": 109930
}
```

## Configuration

### Environment Variables (.env)

```env
# Model Configuration
MODEL_SAVE_PATH=models/digit_recognition.h5
CHECKPOINT_PATH=models/checkpoints
LOG_PATH=logs
MODEL_PATH=models/digit_recognition.h5

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
DEBUG=False
WORKERS=4

# Data Configuration
DATA_PATH=data

# Logging
LOG_LEVEL=INFO
```

### Training Configuration (src/config.py)

Modify `TrainingConfig` class for architecture and training parameters:

```python
# Model architecture
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10
HIDDEN_UNITS_1 = 128
HIDDEN_UNITS_2 = 128
DROPOUT_RATE = 0.2

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t mnist-digit-recognition .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  mnist-digit-recognition
```

### Docker Compose

```bash
# Start services
docker-compose up

# Stop services
docker-compose down

# View logs
docker-compose logs -f mnist-app
```

**Docker Compose features:**
- Automatic image build
- Port mapping (5000:5000)
- Volume mounts for models, data, and logs
- Health checks
- Auto-restart policy

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_model.py -v
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Lint with pylint
pylint src/

# Check import order with isort
isort src/ tests/
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
black src/ tests/
flake8 src/ tests/
pytest --cov=src
```

## Requirements

- Python 3.11+
- TensorFlow 2.13.0
- Keras 2.13.0
- OpenCV 4.8.0
- Flask 2.3.2
- NumPy 1.24.3

See `requirements.txt` for complete dependencies including dev tools.

## Performance Metrics

Expected performance on MNIST test set (10 epochs, default config):

- **Test Accuracy**: ~97-98%
- **Test Loss**: ~0.08-0.10
- **Inference Speed**: ~10-20ms per image (CPU)
- **Model Size**: ~430KB

## Troubleshooting

### Model fails to load
```
Error: Failed to load model
```
**Solution:** Ensure `MODEL_PATH` in `.env` points to a valid trained model file.

### API returns "Model not loaded"
```
Error: Model not loaded
```
**Solution:** Check that the model file exists and API has read permissions. Verify the path in your `.env` file.

### Image preprocessing issues
```
Error: Invalid image format
```
**Solution:** Ensure images are valid grayscale or RGB formats. OpenCV supports PNG, JPG, BMP, etc.

### Out of memory during training
**Solution:** Reduce `BATCH_SIZE` or `HIDDEN_UNITS` in config. Try `--batch-size 16` when training.

### Port already in use
```
Error: Address already in use (:5000)
```
**Solution:** Change `API_PORT` in `.env` or kill the process using port 5000.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST dataset from [Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- Built with [TensorFlow/Keras](https://www.tensorflow.org/)
- OpenCV for image processing
- Flask for REST API

## Support

For issues, questions, or suggestions:

1. Check existing [GitHub Issues](https://github.com/Jayneel467/mnist-digit-recognition/issues)
2. Create a [new issue](https://github.com/Jayneel467/mnist-digit-recognition/issues/new) with details
3. Include error logs and reproduction steps

## Roadmap

- [ ] Convolutional Neural Network (CNN) model variant
- [ ] Batch normalization support
- [ ] Model quantization for edge deployment
- [ ] Web UI for predictions
- [ ] Real-time webcam digit recognition
- [ ] MLflow integration for experiment tracking
- [ ] Kubernetes deployment manifests
- [ ] GPU acceleration guide

---

**Last Updated:** July 2026  
**Status:** Production Ready
