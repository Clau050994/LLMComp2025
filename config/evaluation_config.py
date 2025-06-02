# Evaluation configuration
# Contains settings for model evaluation and benchmarking

# Evaluation metrics and weights
METRICS = {
    "accuracy": {
        "weight": 0.25,
        "description": "Correctness of model outputs"
    },
    "latency": {
        "weight": 0.2,
        "description": "Time taken to return a prediction (ms)",
        "thresholds": {
            "excellent": 100,  # Less than 100ms
            "good": 250,      # Less than 250ms
            "acceptable": 500, # Less than 500ms
            "poor": float('inf')  # More than 500ms
        }
    },
    "model_size": {
        "weight": 0.15,
        "description": "Storage size of the model (MB)"
    },
    "memory_usage": {
        "weight": 0.15,
        "description": "Peak RAM usage during inference (MB)"
    },
    "inference_time": {
        "weight": 0.1,
        "description": "Time for specific tasks (ms)"
    },
    "power_consumption": {
        "weight": 0.1,
        "description": "Estimated power usage during inference (watts)"
    },
    "robustness": {
        "weight": 0.05,
        "description": "Resilience to noise or adversarial inputs"
    }
}

# Edge device constraints
DEVICE_CONSTRAINTS = {
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

# Benchmark tasks
BENCHMARK_TASKS = [
    {
        "name": "document_verification",
        "description": "Verify passport information",
        "input_template": "Verify if the following passport information is valid and consistent:\nName: {name}\nDOB: {dob}\nExpiry: {expiry}\nIssue Date: {issue_date}\nCountry: {country}",
        "expected_outputs": ["valid", "expired", "inconsistent"],
        "metrics": ["accuracy", "latency", "robustness"]
    },
    {
        "name": "multilingual_translation",
        "description": "Translate instructions to travelers",
        "input_template": "Translate to {language}: {text}",
        "languages": ["Spanish", "French", "Arabic", "Russian", "Mandarin", "German"],
        "metrics": ["accuracy", "latency"]
    },
    {
        "name": "entity_detection",
        "description": "Detect potential threats or suspicious entities",
        "input_template": "Analyze if the following travel pattern is suspicious: {pattern}",
        "metrics": ["accuracy", "latency", "robustness"]
    },
    {
        "name": "customs_declaration",
        "description": "Help with customs declarations",
        "input_template": "Explain customs regulations for {item} in simple {language}",
        "metrics": ["accuracy", "helpfulness", "multilingual"]
    },
    {
        "name": "report_generation",
        "description": "Generate structured reports from unstructured inputs",
        "input_template": "Generate a structured report from: {incident_details}",
        "metrics": ["accuracy", "completeness", "structure"]
    }
]

# Scoring function for overall model performance
def calculate_edge_deployment_score(metrics_results):
    """
    Calculate an overall edge deployment score from metric results.
    
    Args:
        metrics_results (dict): Dictionary of metric results
        
    Returns:
        float: Edge deployment score between 0 and 1
    """
    score = 0.0
    
    # Apply weights to each metric
    for metric_name, result in metrics_results.items():
        if metric_name in METRICS:
            weight = METRICS[metric_name]["weight"]
            # Normalize result to 0-1 range (simplified - in a real implementation,
            # this would have more sophisticated normalization logic)
            normalized_result = min(result, 1.0) if result <= 1.0 else 1.0/result
            score += weight * normalized_result
    
    return score
