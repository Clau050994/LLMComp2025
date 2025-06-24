# SLM Evaluation Framework Documentation

This documentation provides an overview of the SLM-SageMaker-Eval project, which evaluates Small Language Models (SLMs) for deployment on edge devices.

## Project Overview

The SLM Eval framework allows you to train, evaluate, and test Small Language Models. The framework provides tools for:

- Evaluating model performance on various metrics
- Optimizing models through quantization
- Benchmarking models for specific use cases
- Testing models with custom inputs

## Project Structure

```
slm-sagemaker-eval/
├── run.py                 # Simple command-line interface to run the framework
├── quick_test.py          # Script for quickly testing models with custom inputs
├── requirements.txt       # Python package dependencies
├── src/                   # Main source code directory
│   ├── main.py            # Core entry point for the application
│   ├── models/            # Model definitions and registry
│   ├── training/          # Code for training models
│   ├── evaluation/        # Code for evaluating models
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── logs/                  # Logs from running the framework
├── results/               # Results from model evaluations
└── LICENSE                # License information
```

## Components Description

### Core Files

#### `run.py`
- Simple wrapper script with user-friendly CLI for running the framework
- Makes it easy to run complex commands with a simplified interface
- Example: `python run.py train --model phi-3-mini --dataset squad`

#### `src/main.py`
- Main entry point for the framework
- Handles command-line arguments and routes to the right functionality
- Coordinates between different modules (training, evaluation, etc.)
- Provides detailed options for customizing model handling

### Model Management

#### `src/models/__init__.py`
- Contains a registry of all available SLMs:
  - Phi-3-mini (128K context)
  - TinyLLaMA-1.1B
  - DistilBERT
  - ALBERT
  - MobileBERT
  - MobileLLaMA
  - Grok-3 mini-fast
- Stores model configurations, including HuggingFace model IDs
- Provides functions to get information about models

### Training System

#### `src/training/__init__.py`
- Provides functions to train models either locally or on AWS SageMaker
- Handles dataset loading and preparation
- Sets up training configurations
- Manages the SageMaker training job creation and monitoring

#### `src/training/scripts/train.py`
- The script that actually runs on SageMaker for training
- Contains the training loop and data processing logic
- Sets up logging, evaluation, and model saving
- Handles different dataset types and formats

### Evaluation System

#### `src/evaluation/__init__.py`
- Functions to evaluate models on different metrics:
  - Accuracy
  - Latency
  - Memory usage
  - Model size
  - Inference time
  - Power consumption
  - Robustness
- Simulates target device constraints for realistic measurements

#### `src/evaluation/benchmark.py`
- More detailed benchmarking for models
- Runs standardized tests for border control use cases
- Includes tests for:
  - Document verification
  - Multilingual translation
  - Entity and threat detection
  - Customs declaration help
  - Mobile report generation
- Generates comprehensive performance reports with overall scores

### Quick Testing System

#### `quick_test.py`
- Enables quick testing of models with custom inputs
- Supports both encoder-only models and generative models
- Provides human-readable output for model responses
- Helps debug and verify model behavior before full evaluation

### Utilities

#### `src/utils/quantization.py`
- Functions to quantize models (reduce their size by lowering precision)
- Supports 4-bit and 8-bit quantization
- Validates whether models are suitable for quantization
- Optimizes models for edge deployment

#### `src/utils/conversion.py`
- Converts models to ONNX format for better edge performance
- Optimizes models for specific hardware targets
- Simplifies model computational graphs where possible
- Preserves tokenizer and configuration along with the model

#### `src/utils/logger.py`
- Sets up logging for the application
- Configures console and file handlers
- Helps track what's happening during training and evaluation
- Includes rotation functionality for log files

### Data Directory

#### `data/models/`
- Stores saved model files after training
- Organized by model name and version
- Contains both full and quantized model variants
- Includes ONNX-exported models for edge deployment

#### Jupyter notebooks for analysis and experimentation
- Model comparison and visualization notebooks
- Performance analysis across different edge devices
- Parameter tuning and optimization notebooks
- Results visualization and reporting notebooks

### Config Directory

#### `config/config.json`
- Main configuration file with default settings
- Contains global parameters for the entire framework
- Defines logging levels, output paths, and default behaviors

#### `config/evaluation_config.py`
- Configuration parameters for model evaluation
- Defines metrics, thresholds, and test scenarios
- Contains device simulation parameters for edge testing

#### `config/sagemaker_config.py`
- AWS SageMaker specific configuration
- Contains S3 bucket information, IAM roles, and regions
- Defines instance types for different models
- Specifies framework versions for SageMaker training

## Using the Framework

The framework is designed to follow this workflow:

This will detect your platform, check for required dependencies, and provide specific recommendations for your environment.

### 1. Train a model

Train a model using SageMaker or locally:

```bash
# Train locally
python run.py train --model phi-3-mini --dataset squad
```

### 2. Evaluate model performance

Evaluate the model's performance:

```bash
python run.py evaluate --model phi-3-mini --device docker-sim --metrics accuracy latency model_size
```

### 3. Quick test a model

Run a quick test on a model with custom input:


### 4. Optimize the model

Quantize the model to make it smaller:

```bash
python run.py quantize --model phi-3-mini --bits 4
```

### 5. Run benchmarks

Benchmark the model on specific use cases:

```bash
python run.py benchmark --model phi-3-mini --device docker-sim
```

## Model Evaluation Parameters

When evaluating SLMs for edge deployment, the framework focuses on these key parameters:

1. **Accuracy** - Measures the correctness of model outputs (e.g., exact match, F1 score)
2. **Latency** - Time taken by the model to return a prediction (in milliseconds)
3. **Model Size** - The storage footprint of the model (in MB or GB)
4. **Memory Usage** - Peak RAM required during inference
5. **Inference Time** - Time needed for processing specific border control tasks
6. **Power Consumption** - Estimated energy usage during inference
7. **Robustness** - Model's resilience to noise, typos, or adversarial inputs

## Border Control Use Cases

The framework includes specific evaluation scenarios for border control:

1. **On-the-Spot Document Verification** - Scanning passports, IDs, or visas
2. **Conversational Triage in Multiple Languages** - Multilingual communication
3. **Entity and Threat Detection** - Flagging suspicious patterns
4. **Customs Declaration Help** - Providing rules and explanations
5. **Mobile Report Generation** - Creating structured reports from incidents

## Prerequisites

To use the framework, you need:

1. Python 3.8 or higher
2. PyTorch
3. HuggingFace Transformers
4. AWS S3 buckets
5. Docker (for simulated deployments)
6. ONNX Runtime (for model conversion)

## Installation

### Basic Installation

Install requirements:

```bash
pip install -r requirements.txt
```

Optional packages for specific features:

```bash
# For model quantization
pip install bitsandbytes

# For ONNX conversion
pip install onnx onnxruntime onnxsim
```

### Windows-Specific Instructions

The framework is designed to be cross-platform but requires some additional setup on Windows:

1. **Path Handling**: The framework uses Python's `os.path` module which should handle path differences automatically. However, if you encounter path issues, verify that paths are constructed using `os.path.join()` rather than hardcoded slashes.

2. **Package Installation**:
   - For `bitsandbytes` on Windows, you may need the specific Windows version:
   ```bash
   pip install bitsandbytes-windows
   ```
   - Alternatively, consider using WSL (Windows Subsystem for Linux) for a more Linux-like environment.

3. **Docker Configuration**:
   - On Windows, use Docker Desktop with WSL 2 backend for optimal performance.
   - Some Docker commands may require slight syntax adjustments in PowerShell.

4. **CUDA Support**:
   - If using NVIDIA GPUs on Windows, ensure you have the appropriate CUDA toolkit and drivers installed.
   - PyTorch installation with CUDA support on Windows may require a specific command:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (replace cu118 with your CUDA version)

5. **Shell Commands**:
   - Some deployment scripts use bash commands. On Windows, consider using Git Bash, WSL, or adapt the commands to PowerShell/CMD equivalents.
   - The framework uses Python's platform detection to adapt terminal commands where needed.




