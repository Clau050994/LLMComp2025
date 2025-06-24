import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.utils.logger import setup_logger

# Setup logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"run_{timestamp}.log")
logger = setup_logger(log_file)

AVAILABLE_MODELS = [
    "phi-3-mini", "tinyllama-1.1b", "distilbert", "albert",
    "mobilebert", "mobilellama", "grok-3-mini-fast"
]

EDGE_DEVICES = ["raspberry-pi", "jetson-nano", "docker-sim"]

METRICS = [
    "accuracy", "latency", "model_size", "memory_usage",
    "inference_time", "power_consumption", "robustness"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='SLM Eval Framework')
    parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS)
    parser.add_argument('--action', type=str, required=True, choices=['train', 'evaluate', 'benchmark', 'quantize'])
    parser.add_argument('--dataset', type=str, default='generic')
    parser.add_argument('--device', type=str, choices=EDGE_DEVICES)
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--metrics', type=str, nargs='+', default=METRICS, choices=METRICS)
    return parser.parse_args()

def main():
    args = parse_arguments()

    logger.info(f"Starting SLM-Eval with model: {args.model}")
    logger.info(f"Action: {args.action}")

    try:
        model_config = get_model(args.model)

        if args.action == 'train':
            logger.info(f"Training model {args.model} on dataset {args.dataset}")
            train_model(
                model_name=args.model,
                model_config=model_config,
                dataset_csv=args.dataset,
                batch_size=args.batch_size,
                epochs=args.epochs,
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

        elif args.action == 'quantize':
            from src.utils.quantization import quantize_model
            logger.info(f"Quantizing model {args.model}")
            quantize_model(
                model_name=args.model,
                model_config=model_config
            )

        elif args.action == 'benchmark':
            if not args.device:
                logger.error("Device must be specified for benchmarking")
                sys.exit(1)
            from src.evaluation.benchmark import run_benchmark
            logger.info(f"Benchmarking model {args.model} on device {args.device}")
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
