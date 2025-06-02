# SageMaker configuration
# Contains settings for SageMaker training and deployment

# S3 bucket for storing model artifacts
S3_BUCKET = "slm-sagemaker-eval"
S3_PREFIX = "models"

# SageMaker execution role
ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Default region
REGION = "us-west-2"

# Hyperparameters for training
DEFAULT_HYPERPARAMETERS = {
    "epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100
}

# Instance types
INSTANCE_TYPES = {
    "phi-3-mini": "ml.g5.2xlarge",
    "tinyllama-1.1b": "ml.g4dn.xlarge",
    "distilbert": "ml.c5.xlarge",
    "albert": "ml.c5.large",
    "mobilebert": "ml.c5.large",
    "mobilellama": "ml.g4dn.xlarge",
    "grok-3-mini-fast": "ml.g5.xlarge",
    "default": "ml.c5.xlarge"
}

# Framework versions
FRAMEWORK_VERSIONS = {
    "transformers": "4.34.0",
    "pytorch": "2.0.0",
    "python": "py39"
}
