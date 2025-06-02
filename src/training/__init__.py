"""
SageMaker training module for the SLM evaluation framework.
This module provides functions for training models on AWS SageMaker.
"""

import os
import logging
from typing import Dict, Any, Optional

import boto3
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFace

logger = logging.getLogger(__name__)

def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    dataset: str,
    batch_size: int = 8,
    epochs: int = 3,
    use_sagemaker: bool = True,
    output_path: Optional[str] = None
) -> str:
    """
    Train the model using AWS SageMaker or locally.
    
    Args:
        model_name (str): Name of the model to train
        model_config (Dict[str, Any]): Model configuration from the registry
        dataset (str): Dataset to train on
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        use_sagemaker (bool): Whether to use SageMaker for training
        output_path (str, optional): Path to save the trained model
        
    Returns:
        str: Path to the trained model
    """
    logger.info(f"Starting training for model {model_name}")
    
    if not use_sagemaker:
        return _train_locally(model_name, model_config, dataset, batch_size, epochs, output_path)
    else:
        return _train_on_sagemaker(model_name, model_config, dataset, batch_size, epochs)

def _train_on_sagemaker(
    model_name: str,
    model_config: Dict[str, Any],
    dataset: str,
    batch_size: int,
    epochs: int
) -> str:
    """
    Train the model on AWS SageMaker.
    
    Args:
        model_name (str): Name of the model to train
        model_config (Dict[str, Any]): Model configuration from the registry
        dataset (str): Dataset to train on
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        
    Returns:
        str: S3 path to the trained model
    """
    try:
        logger.info("Setting up SageMaker training job")
        
        role = get_execution_role()
        
        # Create a session
        sagemaker_session = boto3.Session().client('sagemaker')
        
        # Define hyperparameters
        hyperparameters = {
            'model_id': model_config['hf_model_id'],
            'dataset': dataset,
            'epochs': epochs,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size
        }
        
        # Create HuggingFace estimator
        huggingface_estimator = HuggingFace(
            entry_point='train.py',
            source_dir='src/training/scripts',
            role=role,
            instance_count=1,
            instance_type=model_config['instance_type'],
            transformers_version=model_config['sagemaker_framework_version'],
            pytorch_version='1.13.1',
            py_version='py39',
            hyperparameters=hyperparameters,
            output_path=f's3://slm-sagemaker-eval/models/{model_name}'
        )
        
        # Start training
        logger.info(f"Starting SageMaker training job for {model_name}")
        huggingface_estimator.fit()
        
        # Get the model path
        model_path = huggingface_estimator.model_data
        
        logger.info(f"Training completed. Model saved to {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error in SageMaker training: {str(e)}", exc_info=True)
        raise
    
def _train_locally(
    model_name: str,
    model_config: Dict[str, Any],
    dataset: str,
    batch_size: int,
    epochs: int,
    output_path: Optional[str] = None
) -> str:
    """
    Train the model locally (for development or testing).
    
    Args:
        model_name (str): Name of the model to train
        model_config (Dict[str, Any]): Model configuration from the registry
        dataset (str): Dataset to train on
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        output_path (str, optional): Path to save the trained model
        
    Returns:
        str: Path to the trained model
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        import datasets
        
        logger.info(f"Starting local training for model {model_name}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = f"models/{model_name}/local/{epochs}epochs"
            os.makedirs(output_path, exist_ok=True)
        
        # Load the dataset
        logger.info(f"Loading dataset {dataset}")
        
        # For demo purposes, we'll use a simple public dataset
        # In a real scenario, you would load your specific dataset
        if dataset == "generic":
            dataset = "glue/sst2"
        
        # Load dataset
        train_dataset = datasets.load_dataset(dataset, split="train")
        eval_dataset = datasets.load_dataset(dataset, split="validation")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_config["hf_model_id"])
        model = AutoModelForSequenceClassification.from_pretrained(model_config["hf_model_id"])
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch",
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info(f"Starting local training for {model_name}")
        trainer.train()
        
        # Save model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"Training completed. Model saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in local training: {str(e)}", exc_info=True)
        raise
