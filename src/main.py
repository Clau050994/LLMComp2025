#!/usr/bin/env python3
"""
SLM-SageMaker-Eval: Main entry point for model training and evaluation.
This script provides a command-line interface to train and evaluate various 
Small Language Models (SLMs) on AWS SageMaker for edge deployment.

Usage:
    python main.py --model MODEL_NAME --action ACTION [--dataset DATASET] [--device DEVICE] 
                  [--quantize] [--batch_size BATCH_SIZE] [--epochs EPOCHS]

Example:
    python main.py --model phi-3-mini --action train --dataset squad
    python main.py --model tinyllama --action evaluate --device raspberry-pi --quantize
"""

import argparse
import logging
import os
import sys
import platform
from datetime import datetime

# Import platform utilities for cross-platform compatibility
from utils.platform_utils import setup_platform_specific_configs, get_platform_name, normalize_path

# Add src to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.deployment import deploy_model
from src.utils.logger import setup_logger

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/run_{timestamp}.log"
os.makedirs('logs', exist_ok=True)
logger = setup_logger(log_file)

# Available models
AVAILABLE_MODELS = [
    "phi-3-mini",
    "tinyllama-1.1b",
    "distilbert",
    "albert",
    "mobilebert",
    "mobilellama",
    "grok-3-mini-fast"
]

# Available edge devices
EDGE_DEVICES = [
    "raspberry-pi",
    "jetson-nano", 
    "docker-sim"
]

# Available evaluation metrics
METRICS = [
    "accuracy",
    "latency",
    "model_size",
    "memory_usage",
    "inference_time",
    "power_consumption",
    "robustness"
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate Small Language Models on AWS SageMaker for edge deployment.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=AVAILABLE_MODELS,
        help='Name of the model to train/evaluate'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'deploy', 'benchmark', 'quantize'],
        help='Action to perform on the model'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='generic',
        help='Dataset to use for training or evaluation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=EDGE_DEVICES,
        help='Target edge device for evaluation/deployment'
    )
    
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Whether to quantize the model'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training/evaluation'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs for training'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=METRICS,
        choices=METRICS,
        help='Metrics to evaluate'
    )
    
    parser.add_argument(
        '--sagemaker',
        action='store_true',
        help='Whether to use SageMaker for training/evaluation'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the SLM evaluation framework."""
    args = parse_arguments()
    
    # Set up platform-specific configurations
    setup_platform_specific_configs()
    logger.info(f"Starting SLM-SageMaker-Eval with model: {args.model}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Running on platform: {get_platform_name()}")
    
    try:
        # Get the specified model
        model_config = get_model(args.model)
        
        # Perform the specified action
        if args.action == 'train':
            logger.info(f"Training model {args.model} on dataset {args.dataset}")
            train_model(
                model_name=args.model,
                model_config=model_config,
                dataset=args.dataset,
                batch_size=args.batch_size,
                epochs=args.epochs,
                use_sagemaker=args.sagemaker
            )
            
        elif args.action == 'evaluate':
            if not args.device:
                logger.error("Device must be specified for evaluation")
                sys.exit(1)
                
            logger.info(f"Evaluating model {args.model} on device {args.device}")
            evaluate_model(
                model_name=args.model,
                model_config=model_config,
                device=args.device,
                metrics=args.metrics,
                quantized=args.quantize
            )
            
        elif args.action == 'deploy':
            if not args.device:
                logger.error("Device must be specified for deployment")
                sys.exit(1)
                
            logger.info(f"Deploying model {args.model} to {args.device}")
            deploy_model(
                model_name=args.model,
                model_config=model_config,
                device=args.device,
                quantized=args.quantize
            )
            
        elif args.action == 'quantize':
            logger.info(f"Quantizing model {args.model}")
            from src.utils.quantization import quantize_model
            quantize_model(
                model_name=args.model,
                model_config=model_config
            )
            
        elif args.action == 'benchmark':
            if not args.device:
                logger.error("Device must be specified for benchmarking")
                sys.exit(1)
                
            logger.info(f"Benchmarking model {args.model} on device {args.device}")
            from src.evaluation.benchmark import run_benchmark
            run_benchmark(
                model_name=args.model,
                model_config=model_config,
                device=args.device,
                metrics=args.metrics,
                quantized=args.quantize
            )
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("Execution completed successfully")

if __name__ == "__main__":
    main()
