"""
Model registry for SLM evaluation framework.
This module defines the available models and their configurations.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Model registry with configurations
MODEL_REGISTRY = {
    "phi-3-mini": {
        "name": "Phi-3-mini",
        "hf_model_id": "microsoft/phi-3-mini",
        "context_length": 128000,
        "parameters": "3.8B",
        "description": "Small but powerful model with 128K context from Microsoft",
        "quantization_supported": True,
        "onnx_supported": True,
    },
    "tinyllama-1.1b": {
        "name": "TinyLLaMA-1.1B",
        "hf_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "context_length": 2048,
        "parameters": "1.1B",
        "description": "Compact generative model with 1.1B parameters",
        "quantization_supported": True,
        "onnx_supported": True,
    },
    "distilbert": {
        "name": "DistilBERT",
        "hf_model_id": "distilbert-base-uncased",
        "context_length": 512,
        "parameters": "66M",
        "description": "Distilled version of BERT with 40% fewer parameters",
        "quantization_supported": True,
        "onnx_supported": True,
    },
    "albert": {
        "name": "ALBERT",
        "hf_model_id": "albert-base-v2",
        "context_length": 512,
        "parameters": "12M",
        "description": "A Lite BERT with parameter reduction techniques",
        "quantization_supported": True,
        "onnx_supported": True,
    },
    "mobilebert": {
        "name": "MobileBERT",
        "hf_model_id": "google/mobilebert-uncased",
        "context_length": 512,
        "parameters": "25M",
        "description": "BERT optimized for mobile devices with bottleneck structures",
        "quantization_supported": True,
        "onnx_supported": True,
    },
    "mobilellama": {
        "name": "MobileLLaMA",
        "hf_model_id": "kiri-ai/MobileLLaMA-1.4B-Chat",
        "context_length": 2048,
        "parameters": "1.4B",
        "description": "Optimized version of LLaMA for mobile and edge devices",
        "quantization_supported": True,
        "onnx_supported": True,
    },
    "grok-3-mini-fast": {
        "name": "Grok-3-mini-fast",
        "hf_model_id": "xai-org/grok-3-mini-fast",
        "context_length": 8192,
        "parameters": "3B",
        "description": "Fast version of Grok-3-mini with good performance-size trade-off",
        "quantization_supported": True,
        "onnx_supported": True,
    }
}

def get_model(model_name: str) -> Dict[str, Any]:
    """
    Get model configuration by name.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Model configuration
        
    Raises:
        ValueError: If model is not found
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Model {model_name} not found in registry")
        raise ValueError(f"Model {model_name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    logger.info(f"Retrieved configuration for model {model_name}")
    return MODEL_REGISTRY[model_name]

def list_available_models():
    """List all available models in the registry."""
    return list(MODEL_REGISTRY.keys())

def get_model_details(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Detailed model information
    """
    model_config = get_model(model_name)
    return {
        "name": model_config["name"],
        "parameters": model_config["parameters"],
        "context_length": model_config["context_length"],
        "description": model_config["description"],
        "quantization_supported": model_config["quantization_supported"],
        "onnx_supported": model_config["onnx_supported"]
    }
