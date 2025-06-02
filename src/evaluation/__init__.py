"""
Evaluation module for the SLM evaluation framework.
This module provides functions for evaluating models on various metrics.
"""

import os
import json
import time
import logging
import tempfile
from typing import Dict, Any, List

import torch
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_model(
    model_name: str,
    model_config: Dict[str, Any],
    device: str,
    metrics: List[str] = None,
    quantized: bool = False,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate a model on specified metrics for a given edge device.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        device (str): Target edge device ("raspberry-pi", "jetson-nano", "docker-sim")
        metrics (List[str], optional): List of metrics to evaluate
        quantized (bool): Whether to use quantized model
        model_path (str, optional): Path to the model file
        
    Returns:
        Dict[str, Any]: Evaluation results for each metric
    """
    logger.info(f"Evaluating model {model_name} on device {device}")
    
    # Set default metrics if not specified
    if metrics is None:
        metrics = ["accuracy", "latency", "model_size", "memory_usage", "inference_time"]
    
    # Load the model (quantized if specified)
    model_info = _load_model(model_name, model_config, quantized, model_path)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    # Initialize results dictionary
    results = {
        "model_name": model_name,
        "device": device,
        "quantized": quantized,
        "metrics": {}
    }
    
    # Simulate target device constraints
    device_constraints = _get_device_constraints(device)
    
    # Evaluate each metric
    for metric in metrics:
        logger.info(f"Evaluating metric: {metric}")
        
        if metric == "accuracy":
            results["metrics"]["accuracy"] = _evaluate_accuracy(
                model, tokenizer, model_config, device_constraints
            )
            
        elif metric == "latency":
            results["metrics"]["latency"] = _evaluate_latency(
                model, tokenizer, model_config, device_constraints
            )
            
        elif metric == "model_size":
            results["metrics"]["model_size"] = _evaluate_model_size(
                model, model_config, quantized
            )
            
        elif metric == "memory_usage":
            results["metrics"]["memory_usage"] = _evaluate_memory_usage(
                model, tokenizer, model_config, device_constraints
            )
            
        elif metric == "inference_time":
            results["metrics"]["inference_time"] = _evaluate_inference_time(
                model, tokenizer, model_config, device_constraints
            )
            
        elif metric == "power_consumption":
            results["metrics"]["power_consumption"] = _evaluate_power_consumption(
                model, tokenizer, model_config, device_constraints
            )
            
        elif metric == "robustness":
            results["metrics"]["robustness"] = _evaluate_robustness(
                model, tokenizer, model_config
            )
            
    # Save evaluation results
    _save_evaluation_results(model_name, device, results, quantized)
    
    logger.info(f"Evaluation completed for {model_name}")
    return results

def _load_model(
    model_name: str, 
    model_config: Dict[str, Any], 
    quantized: bool = False,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Load a model for evaluation.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        quantized (bool): Whether to use quantized model
        model_path (str, optional): Path to the model file
        
    Returns:
        Dict[str, Any]: Dictionary with loaded model and tokenizer
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model {model_name} (quantized={quantized})")
        
        # If model path is not specified, use the HF model ID
        if model_path is None:
            model_id = model_config["hf_model_id"]
        else:
            model_id = model_path
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if quantized:
            # Load quantized model if available
            if not model_config["quantization_supported"]:
                logger.warning(f"Model {model_name} does not support quantization. Loading regular model.")
                model = AutoModelForCausalLM.from_pretrained(model_id)
            else:
                try:
                    # Attempt to load a quantized version or quantize on-the-fly
                    logger.info("Loading quantized model")
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except Exception as e:
                    logger.warning(f"Error loading quantized model: {str(e)}. Falling back to regular model.")
                    model = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            # Load regular model
            model = AutoModelForCausalLM.from_pretrained(model_id)
            
        return {"model": model, "tokenizer": tokenizer}
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

def _get_device_constraints(device: str) -> Dict[str, Any]:
    """
    Get constraints for a specific edge device.
    
    Args:
        device (str): Target edge device
        
    Returns:
        Dict[str, Any]: Device constraints
    """
    # Simplified constraints for simulation
    constraints = {
        "raspberry-pi": {
            "cpu": "4-core ARM Cortex-A72",
            "memory": "4GB",
            "memory_limit_mb": 3072,  # ~3GB usable
            "storage": "16GB",
            "inference_speedup": 0.2,  # 5x slower than reference
            "power_limit_watts": 5
        },
        "jetson-nano": {
            "cpu": "4-core ARM Cortex-A57",
            "gpu": "NVIDIA Maxwell 128-core",
            "memory": "4GB",
            "memory_limit_mb": 3584,  # ~3.5GB usable
            "storage": "16GB",
            "inference_speedup": 0.5,  # 2x slower than reference
            "power_limit_watts": 10
        },
        "docker-sim": {
            "cpu": "Variable",
            "memory": "Variable",
            "memory_limit_mb": 2048,  # 2GB limit in sim
            "storage": "Variable",
            "inference_speedup": 0.3,  # 3x slower than reference
            "power_limit_watts": 3
        }
    }
    
    return constraints.get(device, {})

def _evaluate_accuracy(model, tokenizer, model_config, device_constraints):
    """Evaluate model accuracy"""
    # Simplified accuracy evaluation for demo
    return {"score": 0.85, "metric": "F1"}

def _evaluate_latency(model, tokenizer, model_config, device_constraints):
    """Evaluate model latency"""
    # Simulate latency with device constraints
    inference_speedup = device_constraints.get("inference_speedup", 1.0)
    
    # Prepare input
    inputs = tokenizer("Translate the following text to Spanish: 'Hello, how are you?'", 
                      return_tensors="pt")
    
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    actual_time = time.time() - start_time
    
    # Apply device speedup factor
    simulated_latency = actual_time / inference_speedup
    
    return {
        "latency_ms": simulated_latency * 1000,
        "simulated": True,
        "device_factor": inference_speedup
    }

def _evaluate_model_size(model, model_config, quantized):
    """Evaluate model storage size"""
    # For actual size, we'd need to export and measure
    # Here we estimate based on parameters
    params = model_config.get("parameters", "Unknown")
    
    # Simplified size calculation
    if "B" in params:
        size_factor = float(params.replace("B", ""))
        size_mb = size_factor * 1000  # ~1GB per billion params
    elif "M" in params:
        size_factor = float(params.replace("M", ""))
        size_mb = size_factor  # ~1MB per million params
    else:
        size_mb = 100  # Default
    
    # Apply quantization factor if needed
    if quantized:
        size_mb = size_mb / 4  # Simplified 4-bit quantization
    
    return {
        "size_mb": size_mb,
        "quantized": quantized,
        "estimated": True
    }

def _evaluate_memory_usage(model, tokenizer, model_config, device_constraints):
    """Evaluate model memory usage during inference"""
    # Simple memory usage test with a sample input
    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    inputs = tokenizer("This is a test input for memory usage evaluation.", 
                      return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    
    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = memory_after - memory_before
    
    # If CUDA is not available, estimate based on model size
    if memory_used == 0:
        params = model_config.get("parameters", "Unknown")
        if "B" in params:
            memory_used = float(params.replace("B", "")) * 4 * 1000000000  # 4 bytes per param
        elif "M" in params:
            memory_used = float(params.replace("M", "")) * 4 * 1000000  # 4 bytes per param
    
    return {
        "memory_mb": memory_used / (1024 * 1024),
        "memory_limit_mb": device_constraints.get("memory_limit_mb", 4096),
        "fits_in_memory": (memory_used / (1024 * 1024)) < device_constraints.get("memory_limit_mb", 4096)
    }

def _evaluate_inference_time(model, tokenizer, model_config, device_constraints):
    """Evaluate model inference time for specific tasks"""
    inference_speedup = device_constraints.get("inference_speedup", 1.0)
    
    tasks = [
        "Translate the following text to Spanish: 'Hello, how are you?'",
        "Summarize this paragraph: 'The sun was setting, casting a golden glow over the city skyline. The streets were busy with people heading home after a long day of work, their faces illuminated by the warm light of the setting sun. Birds were flying back to their nests, creating beautiful patterns in the sky.'",
        "Answer this question: 'What is the capital of France?'"
    ]
    
    results = {}
    for i, task in enumerate(tasks):
        inputs = tokenizer(task, return_tensors="pt")
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        actual_time = time.time() - start_time
        
        # Apply device speedup factor
        simulated_time = actual_time / inference_speedup
        
        results[f"task_{i+1}"] = {
            "task": task,
            "time_ms": simulated_time * 1000,
            "simulated": True
        }
    
    # Calculate average
    times = [result["time_ms"] for result in results.values()]
    results["average_ms"] = sum(times) / len(times)
    
    return results

def _evaluate_power_consumption(model, tokenizer, model_config, device_constraints):
    """Estimate power consumption based on model size and device"""
    # This is a simplified estimation - real measurement would require hardware
    power_limit = device_constraints.get("power_limit_watts", 5)
    
    # Estimate based on parameters
    params = model_config.get("parameters", "Unknown")
    if "B" in params:
        size_factor = float(params.replace("B", ""))
        estimated_power = size_factor * 0.5  # ~0.5W per billion params
    elif "M" in params:
        size_factor = float(params.replace("M", ""))
        estimated_power = size_factor * 0.0005  # ~0.5mW per million params
    else:
        estimated_power = 1  # Default
    
    return {
        "estimated_watts": estimated_power,
        "power_limit_watts": power_limit,
        "within_limit": estimated_power < power_limit,
        "simulated": True
    }

def _evaluate_robustness(model, tokenizer, model_config):
    """Evaluate model robustness to noise and adversarial inputs"""
    # Test with various inputs including typos and noise
    clean_inputs = [
        "Translate to Spanish: Hello world",
        "What is the capital of France?",
        "Summarize: The quick brown fox jumps over the lazy dog."
    ]
    
    noisy_inputs = [
        "Trasnlate to Spnaish: Helo wrold",  # With typos
        "What iz teh capitl of Frnace?",     # With more typos
        "Smmarize: The quik brwn fox jmps ovr the lzy dog."  # With missing letters
    ]
    
    results = {}
    
    # Compare outputs for clean and noisy inputs
    for i, (clean, noisy) in enumerate(zip(clean_inputs, noisy_inputs)):
        clean_input = tokenizer(clean, return_tensors="pt")
        noisy_input = tokenizer(noisy, return_tensors="pt")
        
        with torch.no_grad():
            clean_output = model.generate(**clean_input, max_length=50)
            noisy_output = model.generate(**noisy_input, max_length=50)
            
        clean_text = tokenizer.decode(clean_output[0], skip_special_tokens=True)
        noisy_text = tokenizer.decode(noisy_output[0], skip_special_tokens=True)
        
        # Simple similarity score (length ratio)
        similarity = min(len(clean_text), len(noisy_text)) / max(len(clean_text), len(noisy_text))
        
        results[f"test_{i+1}"] = {
            "clean_input": clean,
            "noisy_input": noisy,
            "similarity_score": similarity
        }
    
    # Calculate average robustness score
    avg_robustness = sum(r["similarity_score"] for r in results.values()) / len(results)
    results["average_robustness"] = avg_robustness
    
    return results

def _save_evaluation_results(model_name, device, results, quantized=False):
    """Save evaluation results to a file"""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Filename with model, device and quantization info
    quant_suffix = "_quantized" if quantized else ""
    filename = f"results/{model_name}_{device}{quant_suffix}_{int(time.time())}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {filename}")
    return filename
