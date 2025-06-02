#!/usr/bin/env python3
"""
SLM-SageMaker-Eval: Command-line interface for running the framework.
This is a wrapper around the main script with simplified arguments.
"""

import os
import sys
import argparse
import logging
import platform
from datetime import datetime

# Add src to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.main import (
    AVAILABLE_MODELS,
    EDGE_DEVICES,
    METRICS,
    parse_arguments as main_parse_arguments,
    main as main_function
)

def main():
    """Simple CLI for the SLM evaluation framework."""
    parser = argparse.ArgumentParser(
        description='Small Language Model (SLM) evaluation framework for edge deployment'
    )
    
    # Define command groups
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Add system check command
    check_parser = subparsers.add_parser('check', help='Run system compatibility check')
    check_parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model on SageMaker')
    train_parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS,
                             help='Model to train')
    train_parser.add_argument('--dataset', type=str, default='generic',
                             help='Dataset to use for training')
    train_parser.add_argument('--batch-size', type=int, default=8,
                             help='Training batch size')
    train_parser.add_argument('--epochs', type=int, default=3,
                             help='Number of training epochs')
    train_parser.add_argument('--sagemaker', action='store_true',
                             help='Use SageMaker for training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS,
                            help='Model to evaluate')
    eval_parser.add_argument('--device', type=str, required=True, choices=EDGE_DEVICES,
                            help='Target edge device for evaluation')
    eval_parser.add_argument('--metrics', type=str, nargs='+', choices=METRICS,
                            default=METRICS, help='Metrics to evaluate')
    eval_parser.add_argument('--quantize', action='store_true',
                            help='Use quantized model')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy a model to an edge device')
    deploy_parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS,
                              help='Model to deploy')
    deploy_parser.add_argument('--device', type=str, required=True, choices=EDGE_DEVICES,
                              help='Target edge device for deployment')
    deploy_parser.add_argument('--quantize', action='store_true',
                              help='Deploy quantized model')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark a model')
    benchmark_parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS,
                                 help='Model to benchmark')
    benchmark_parser.add_argument('--device', type=str, required=True, choices=EDGE_DEVICES,
                                 help='Target edge device for benchmarking')
    benchmark_parser.add_argument('--metrics', type=str, nargs='+', choices=METRICS,
                                 default=METRICS, help='Metrics to benchmark')
    benchmark_parser.add_argument('--quantize', action='store_true',
                                 help='Benchmark quantized model')
    
    # Quantize command
    quantize_parser = subparsers.add_parser('quantize', help='Quantize a model')
    quantize_parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS,
                                help='Model to quantize')
    quantize_parser.add_argument('--bits', type=int, choices=[4, 8], default=4,
                                help='Quantization bits (4 or 8)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    # Handle system check command
    if args.command == 'check':
        from src.utils.system_check import run_system_check, print_check_results
        results = run_system_check()
        print_check_results(results)
        sys.exit(0)
    
    # Map the CLI arguments to main script arguments
    main_args = []
    main_args.append('--model')
    main_args.append(args.model)
    
    main_args.append('--action')
    main_args.append(args.command)
    
    if hasattr(args, 'device') and args.device:
        main_args.append('--device')
        main_args.append(args.device)
    
    if hasattr(args, 'dataset') and args.dataset:
        main_args.append('--dataset')
        main_args.append(args.dataset)
    
    if hasattr(args, 'batch_size') and args.batch_size:
        main_args.append('--batch_size')
        main_args.append(str(args.batch_size))
    
    if hasattr(args, 'epochs') and args.epochs:
        main_args.append('--epochs')
        main_args.append(str(args.epochs))
    
    if hasattr(args, 'metrics') and args.metrics:
        main_args.append('--metrics')
        main_args.extend(args.metrics)
    
    if hasattr(args, 'quantize') and args.quantize:
        main_args.append('--quantize')
    
    if hasattr(args, 'sagemaker') and args.sagemaker:
        main_args.append('--sagemaker')
    
    # Parse arguments for main script
    sys.argv = [sys.argv[0]] + main_args
    main_function()

if __name__ == "__main__":
    main()
