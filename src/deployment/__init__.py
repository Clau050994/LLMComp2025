"""
Deployment module for the SLM evaluation framework.
This module provides functions for deploying models to edge devices.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def deploy_model(
    model_name: str,
    model_config: Dict[str, Any],
    device: str,
    quantized: bool = False,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Deploy a model to a specific edge device.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        device (str): Target edge device
        quantized (bool): Whether to deploy a quantized model
        model_path (str, optional): Path to the model file
        
    Returns:
        Dict[str, Any]: Deployment results
    """
    logger.info(f"Deploying model {model_name} to {device} (quantized={quantized})")
    
    # Select appropriate deployment function based on device
    if device == "raspberry-pi":
        return _deploy_to_raspberry_pi(model_name, model_config, quantized, model_path)
    elif device == "jetson-nano":
        return _deploy_to_jetson_nano(model_name, model_config, quantized, model_path)
    elif device == "docker-sim":
        return _deploy_to_docker(model_name, model_config, quantized, model_path)
    else:
        raise ValueError(f"Unsupported device: {device}")

def _deploy_to_raspberry_pi(
    model_name: str,
    model_config: Dict[str, Any],
    quantized: bool,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Deploy model to Raspberry Pi using AWS IoT Greengrass.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        quantized (bool): Whether to use quantized model
        model_path (str, optional): Path to the model file
        
    Returns:
        Dict[str, Any]: Deployment information
    """
    logger.info(f"Setting up Raspberry Pi deployment for {model_name}")
    
    # Prepare model for deployment
    prepared_model = _prepare_model_for_edge(model_name, model_config, quantized, model_path, target="raspberry-pi")
    
    # Create Greengrass component definition
    component_name = f"com.slm.models.{model_name.replace('-', '_')}"
    component_version = "1.0.0"
    
    # Generate Greengrass component recipe
    recipe = {
        "RecipeFormatVersion": "2020-01-25",
        "ComponentName": component_name,
        "ComponentVersion": component_version,
        "ComponentDescription": f"SLM model: {model_name} for edge inference",
        "ComponentPublisher": "SLM-Eval-Project",
        "ComponentConfiguration": {
            "DefaultConfiguration": {
                "modelPath": "{artifacts:path}/model",
                "quantized": quantized
            }
        },
        "Manifests": [
            {
                "Platform": {
                    "os": "linux",
                    "architecture": "arm"
                },
                "Artifacts": [
                    {
                        "URI": f"s3://slm-sagemaker-eval/models/{model_name}/edge/{component_version}/model.tar.gz",
                        "Unarchive": "ZIP"
                    },
                    {
                        "URI": f"s3://slm-sagemaker-eval/models/{model_name}/edge/{component_version}/inference.py"
                    }
                ],
                "Lifecycle": {
                    "Run": "python3 {artifacts:path}/inference.py"
                }
            }
        ]
    }
    
    # In a real scenario, you'd upload artifacts to S3 and create the Greengrass component
    
    # Save recipe for reference
    os.makedirs(f"deployment/{model_name}/greengrass", exist_ok=True)
    with open(f"deployment/{model_name}/greengrass/recipe.json", "w") as f:
        json.dump(recipe, f, indent=2)
    
    logger.info(f"Created Greengrass recipe for {model_name}")
    
    # Return deployment info
    return {
        "status": "prepared",
        "device": "raspberry-pi",
        "model": model_name,
        "quantized": quantized,
        "component_name": component_name,
        "component_version": component_version,
        "recipe_path": f"deployment/{model_name}/greengrass/recipe.json",
        "next_steps": [
            "Upload model artifacts to S3",
            "Create Greengrass component using AWS CLI or console",
            "Deploy component to Raspberry Pi device group"
        ]
    }

def _deploy_to_jetson_nano(
    model_name: str,
    model_config: Dict[str, Any],
    quantized: bool,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Deploy model to NVIDIA Jetson Nano using ONNX and TensorRT.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        quantized (bool): Whether to use quantized model
        model_path (str, optional): Path to the model file
        
    Returns:
        Dict[str, Any]: Deployment information
    """
    logger.info(f"Setting up Jetson Nano deployment for {model_name}")
    
    # First, convert model to ONNX format
    if not model_config["onnx_supported"]:
        logger.warning(f"Model {model_name} does not support ONNX export. Deployment may not be optimal.")
    
    # Prepare model for deployment
    prepared_model = _prepare_model_for_edge(model_name, model_config, quantized, model_path, target="jetson-nano")
    
    # Generate TensorRT optimization script
    tensorrt_script = """
#!/usr/bin/env python3
import os
import tensorrt as trt
import onnx

# Load ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Create TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder:
    with builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:
        with trt.OnnxParser(network, TRT_LOGGER) as parser:
            parser.parse(onnx_model.SerializeToString())
            
            with builder.create_builder_config() as config:
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
                serialized_engine = builder.build_serialized_network(network, config)
                
                with open("model.engine", "wb") as f:
                    f.write(serialized_engine)
                    
print("TensorRT engine created successfully!")
"""
    
    # Save script for reference
    os.makedirs(f"deployment/{model_name}/jetson", exist_ok=True)
    with open(f"deployment/{model_name}/jetson/optimize_model.py", "w") as f:
        f.write(tensorrt_script)
        
    # Create sample inference script
    inference_script = """
#!/usr/bin/env python3
import argparse
import json
import time
import torch
import onnxruntime
import numpy as np
from transformers import AutoTokenizer

def load_model(model_path, use_tensorrt=True):
    if use_tensorrt:
        # Load TensorRT engine
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(model_path, "rb") as f:
            engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            
        context = engine.create_execution_context()
        return {"engine": engine, "context": context}
    else:
        # Load ONNX model
        session = onnxruntime.InferenceSession(model_path)
        return {"session": session}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--use_tensorrt", action="store_true")
    args = parser.parse_args()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args.model, args.use_tensorrt)
    
    # Tokenize input
    inputs = tokenizer(args.input, return_tensors="pt")
    
    # Run inference
    start_time = time.time()
    
    # Different inference paths for TensorRT vs ONNX
    if args.use_tensorrt:
        # TensorRT inference code here
        # (This is a placeholder - actual implementation would depend on model architecture)
        pass
    else:
        # ONNX inference
        ort_inputs = {k: v.numpy() for k, v in inputs.items()}
        ort_outputs = model["session"].run(None, ort_inputs)
        outputs = torch.tensor(ort_outputs[0])
    
    inference_time = time.time() - start_time
    
    # Process outputs (depends on model type)
    # This is a simplified example
    result = {"output": outputs.tolist(), "inference_time_ms": inference_time * 1000}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
    
    with open(f"deployment/{model_name}/jetson/inference.py", "w") as f:
        f.write(inference_script)
    
    # Create deployment instructions
    instructions = f"""
# Deployment Instructions for {model_name} on Jetson Nano

## Prerequisites
- NVIDIA Jetson Nano with JetPack 4.6 or later
- Python 3.8+
- ONNX Runtime
- TensorRT

## Setup Steps
1. Copy model files to the Jetson Nano:
   scp -r deployment/{model_name}/jetson user@jetson-ip:/home/user/{model_name}

2. Install requirements:
   pip install torch torchvision transformers onnx onnxruntime-gpu

3. Convert model to TensorRT engine:
   cd /home/user/{model_name}
   python optimize_model.py

4. Run inference:
   python inference.py --model model.engine --tokenizer tokenizer --input "Your input text" --use_tensorrt

## Performance Notes
- Model size: {model_config.get('parameters', 'N/A')}
- Quantized: {quantized}
- Expected memory usage: See evaluation results
"""
    
    with open(f"deployment/{model_name}/jetson/README.md", "w") as f:
        f.write(instructions)
    
    logger.info(f"Created Jetson Nano deployment package for {model_name}")
    
    return {
        "status": "prepared",
        "device": "jetson-nano",
        "model": model_name,
        "quantized": quantized,
        "files": [
            f"deployment/{model_name}/jetson/optimize_model.py",
            f"deployment/{model_name}/jetson/inference.py",
            f"deployment/{model_name}/jetson/README.md"
        ],
        "next_steps": [
            "Export model to ONNX format (see utils/conversion.py)",
            "Copy deployment files to Jetson Nano",
            "Follow instructions in README.md to complete deployment"
        ]
    }

def _deploy_to_docker(
    model_name: str,
    model_config: Dict[str, Any],
    quantized: bool,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Deploy model to Docker for simulated edge environment.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        quantized (bool): Whether to use quantized model
        model_path (str, optional): Path to the model file
        
    Returns:
        Dict[str, Any]: Deployment information
    """
    logger.info(f"Setting up Docker deployment for {model_name}")
    
    # Prepare model for deployment
    prepared_model = _prepare_model_for_edge(model_name, model_config, quantized, model_path, target="docker")
    
    # Create Dockerfile
    dockerfile = f"""
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV MODEL_NAME={model_name}
ENV QUANTIZED={'true' if quantized else 'false'}

# Install dependencies
RUN pip install --no-cache-dir torch transformers numpy flask gunicorn

# Copy model files
COPY ./model /app/model
COPY ./inference.py /app/inference.py
COPY ./api.py /app/api.py

# Expose API port
EXPOSE 8000

# Run API server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api:app"]
"""
    
    # Create API server script
    api_script = """
from flask import Flask, request, jsonify
import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load model and tokenizer
model_name = os.environ.get('MODEL_NAME', 'model')
quantized = os.environ.get('QUANTIZED', 'false').lower() == 'true'

print(f"Loading model {model_name} (quantized={quantized})")

tokenizer = AutoTokenizer.from_pretrained('./model')
if quantized:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        './model',
        quantization_config=quantization_config,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained('./model')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Return some basic metrics
    return jsonify({
        "model_name": model_name,
        "quantized": quantized,
        "uptime": time.time() - start_time
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 50)
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        inference_time = time.time() - start_time
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "input": text,
            "output": output_text,
            "inference_time_ms": inference_time * 1000
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

start_time = time.time()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
"""
    
    # Create sample inference script
    inference_script = """
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input text for inference")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum output length")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model")
    args = parser.parse_args()
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('./model')
    
    if args.quantized:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            './model',
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained('./model')
        
    print(f"Running inference on input: {args.input}")
    inputs = tokenizer(args.input, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=args.max_length)
    inference_time = time.time() - start_time
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Output: {output_text}")
    print(f"Inference time: {inference_time * 1000:.2f} ms")

if __name__ == "__main__":
    main()
"""
    
    # Create Docker Compose file
    docker_compose = """
version: '3'

services:
  slm-inference:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
"""
    
    # Save files
    os.makedirs(f"deployment/{model_name}/docker", exist_ok=True)
    
    with open(f"deployment/{model_name}/docker/Dockerfile", "w") as f:
        f.write(dockerfile)
        
    with open(f"deployment/{model_name}/docker/api.py", "w") as f:
        f.write(api_script)
        
    with open(f"deployment/{model_name}/docker/inference.py", "w") as f:
        f.write(inference_script)
        
    with open(f"deployment/{model_name}/docker/docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    # Create instructions
    instructions = f"""
# Docker Deployment for {model_name} SLM

This package contains everything needed to deploy the {model_name} model in a
Docker container that simulates edge device constraints.

## Prerequisites
- Docker and Docker Compose
- At least 2GB of RAM available for the container

## Setup Steps

1. Export your model to the ./model directory:
   ```
   python -c "from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('{model_config['hf_model_id']}'); tokenizer = AutoTokenizer.from_pretrained('{model_config['hf_model_id']}'); model.save_pretrained('deployment/{model_name}/docker/model'); tokenizer.save_pretrained('deployment/{model_name}/docker/model')"
   ```

2. Build and run the Docker container:
   ```
   cd deployment/{model_name}/docker
   docker-compose up --build
   ```

3. Test the API:
   ```
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Translate this to Spanish: Hello world", "max_length": 50}'
   ```

## Simulating Edge Constraints

The Docker container is configured to use a maximum of 2 CPU cores and 2GB of RAM,
similar to what might be available on an edge device like a Raspberry Pi 4.
"""
    
    with open(f"deployment/{model_name}/docker/README.md", "w") as f:
        f.write(instructions)
    
    logger.info(f"Created Docker deployment package for {model_name}")
    
    return {
        "status": "prepared",
        "device": "docker-sim",
        "model": model_name,
        "quantized": quantized,
        "files": [
            f"deployment/{model_name}/docker/Dockerfile",
            f"deployment/{model_name}/docker/api.py",
            f"deployment/{model_name}/docker/inference.py",
            f"deployment/{model_name}/docker/docker-compose.yml",
            f"deployment/{model_name}/docker/README.md"
        ],
        "next_steps": [
            "Export model to the deployment directory",
            "Build and run the Docker container",
            "Access the API at http://localhost:8000/predict"
        ]
    }

def _prepare_model_for_edge(
    model_name: str,
    model_config: Dict[str, Any],
    quantized: bool,
    model_path: str = None,
    target: str = "generic"
) -> Dict[str, Any]:
    """
    Prepare model for edge deployment (quantize, optimize, etc.)
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        quantized (bool): Whether to use quantized model
        model_path (str, optional): Path to the model file
        target (str): Target device type
    
    Returns:
        Dict[str, Any]: Path and information about prepared model
    """
    # This is a placeholder for actual model preparation logic
    # In a real implementation, this would handle:
    # - Quantization if requested
    # - Format conversion (e.g., ONNX)
    # - Optimization for target hardware
    
    logger.info(f"Preparing model {model_name} for {target} deployment (quantized={quantized})")
    
    # Mock function that would actually prepare the model
    prepared_info = {
        "model_name": model_name,
        "target": target,
        "quantized": quantized,
        "path": f"deployment/{model_name}/{target}/model",
        "format": "native" if target != "jetson-nano" else "onnx"
    }
    
    return prepared_info
