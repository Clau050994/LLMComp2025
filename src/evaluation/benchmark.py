"""
Benchmarking module for detailed performance analysis of SLMs on edge devices.
This module allows running standardized benchmarks across different models.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List
from datetime import datetime

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Border control specific task templates
BORDER_CONTROL_TASKS = [
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

def run_benchmark(
    model_name: str,
    model_config: Dict[str, Any],
    device: str,
    metrics: List[str] = None,
    quantized: bool = False,
    benchmark_type: str = "border_control",
    num_runs: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive benchmarks for a model on a specific edge device.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict[str, Any]): Model configuration
        device (str): Target edge device
        metrics (List[str], optional): Metrics to evaluate
        quantized (bool): Whether to use quantized model
        benchmark_type (str): Type of benchmark to run
        num_runs (int): Number of runs for each task
        verbose (bool): Whether to print verbose output
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    from src.evaluation import _load_model, _get_device_constraints
    
    logger.info(f"Starting {benchmark_type} benchmark for {model_name} on {device}")
    
    # Load model
    model_info = _load_model(model_name, model_config, quantized)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    # Get device constraints
    device_constraints = _get_device_constraints(device)
    
    # Initialize results
    results = {
        "model": model_name,
        "device": device,
        "quantized": quantized,
        "benchmark_type": benchmark_type,
        "timestamp": datetime.now().isoformat(),
        "overall_metrics": {},
        "task_results": {}
    }
    
    if benchmark_type == "border_control":
        tasks = BORDER_CONTROL_TASKS
    else:
        # Default to general benchmark tasks
        tasks = _get_general_benchmark_tasks()
    
    # Run each task
    for task in tasks:
        task_name = task["name"]
        logger.info(f"Running benchmark task: {task_name}")
        
        task_results = _run_task_benchmark(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device_constraints=device_constraints,
            num_runs=num_runs,
            quantized=quantized
        )
        
        results["task_results"][task_name] = task_results
        
        if verbose:
            print(f"Task: {task_name}")
            for metric, value in task_results["metrics"].items():
                print(f"  {metric}: {value}")
            print()
    
    # Calculate overall metrics
    results["overall_metrics"] = _calculate_overall_metrics(results["task_results"])
    
    # Save benchmark results
    timestamp = int(time.time())
    os.makedirs("benchmarks", exist_ok=True)
    filename = f"benchmarks/{model_name}_{device}_{benchmark_type}_{'quantized' if quantized else 'full'}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {filename}")
    
    if verbose:
        print("\nOverall metrics:")
        for metric, value in results["overall_metrics"].items():
            print(f"  {metric}: {value}")
    
    return results

def _get_general_benchmark_tasks():
    """Get general benchmark tasks for language models"""
    return [
        {
            "name": "text_classification",
            "description": "Classify text sentiment",
            "input_template": "Classify the sentiment of the following text: {text}",
            "metrics": ["accuracy", "latency"]
        },
        {
            "name": "text_generation",
            "description": "Generate coherent text",
            "input_template": "Continue the following passage: {text}",
            "metrics": ["quality", "latency", "memory_usage"]
        },
        {
            "name": "question_answering",
            "description": "Answer questions accurately",
            "input_template": "Question: {question}",
            "metrics": ["accuracy", "latency", "robustness"]
        }
    ]

def _run_task_benchmark(model, tokenizer, task, device_constraints, num_runs=5, quantized=False):
    """Run benchmark for a specific task"""
    task_name = task["name"]
    metrics = task.get("metrics", ["accuracy", "latency"])
    
    # Initialize results for this task
    task_results = {
        "name": task_name,
        "description": task.get("description", ""),
        "num_runs": num_runs,
        "metrics": {},
        "raw_data": []
    }
    
    # Generate test inputs based on task type
    test_inputs = _generate_test_inputs(task, num_runs)
    
    # Run the task multiple times
    for i, test_input in enumerate(test_inputs):
        input_text = test_input["input"]
        expected_output = test_input.get("expected_output", None)
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Measure execution time
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        execution_time = time.time() - start_time
        
        # Apply device speedup factor for latency simulation
        inference_speedup = device_constraints.get("inference_speedup", 1.0)
        simulated_time = execution_time / inference_speedup
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Record run data
        run_data = {
            "input": input_text,
            "output": output_text,
            "expected_output": expected_output,
            "execution_time_ms": execution_time * 1000,
            "simulated_time_ms": simulated_time * 1000
        }
        
        # Add accuracy if we have expected output
        if expected_output is not None:
            run_data["accuracy"] = _calculate_accuracy(output_text, expected_output)
        
        task_results["raw_data"].append(run_data)
    
    # Calculate aggregate metrics
    task_results["metrics"] = _calculate_task_metrics(task_results["raw_data"], metrics)
    
    return task_results

def _generate_test_inputs(task, num_runs=5):
    """Generate test inputs for a specific task type"""
    inputs = []
    
    if task["name"] == "document_verification":
        # Generate realistic passport data scenarios
        names = ["John Smith", "Maria Garcia", "Li Wei", "Ahmed Hassan", "Anna Ivanova"]
        countries = ["USA", "Spain", "China", "Egypt", "Russia"]
        
        for i in range(num_runs):
            # Valid passport
            if i % 3 == 0:
                issue_date = "2020-01-15"
                expiry = "2030-01-15"
                expected = "valid"
            # Expired passport
            elif i % 3 == 1:
                issue_date = "2010-01-15"
                expiry = "2020-01-15"
                expected = "expired"
            # Inconsistent data
            else:
                issue_date = "2022-01-15"  # Issue date after expiry
                expiry = "2020-01-15"
                expected = "inconsistent"
                
            input_text = task["input_template"].format(
                name=names[i % len(names)],
                dob=f"19{80 + i % 20}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                expiry=expiry,
                issue_date=issue_date,
                country=countries[i % len(countries)]
            )
            
            inputs.append({
                "input": input_text,
                "expected_output": expected
            })
            
    elif task["name"] == "multilingual_translation":
        phrases = [
            "Please step aside for additional screening.",
            "Can I see your passport and visa?",
            "What is the purpose of your visit?",
            "How long will you be staying?",
            "Please open your luggage for inspection."
        ]
        
        languages = task.get("languages", ["Spanish", "French", "Arabic", "Russian", "Mandarin"])
        
        for i in range(num_runs):
            language = languages[i % len(languages)]
            phrase = phrases[i % len(phrases)]
            
            input_text = task["input_template"].format(
                language=language,
                text=phrase
            )
            
            inputs.append({
                "input": input_text,
                "expected_output": None  # Translation accuracy would need human evaluation
            })
    
    elif task["name"] == "entity_detection":
        patterns = [
            "Traveler visited 5 high-risk countries in the past month with short stays of 1-2 days each.",
            "Business traveler making regular monthly trips between New York and London for the past year.",
            "Tourist visiting for the first time with a 2-week itinerary and hotel reservations.",
            "Traveler with multiple entry stamps but missing exit stamps from previous visits.",
            "Person traveling with family including children, all with return tickets for next week."
        ]
        
        for i in range(num_runs):
            input_text = task["input_template"].format(
                pattern=patterns[i % len(patterns)]
            )
            
            # For simplicity, we're using simple expected outputs
            # In a real scenario, these would be more nuanced
            expected = "suspicious" if i % 5 == 0 or i % 5 == 3 else "normal"
            
            inputs.append({
                "input": input_text,
                "expected_output": expected
            })
    
    elif task["name"] == "customs_declaration":
        items = ["medication", "alcohol", "electronics", "food", "currency"]
        languages = ["simple English", "Spanish", "French", "Arabic", "Russian"]
        
        for i in range(num_runs):
            input_text = task["input_template"].format(
                item=items[i % len(items)],
                language=languages[i % len(languages)]
            )
            
            inputs.append({
                "input": input_text,
                "expected_output": None  # Would need human evaluation
            })
    
    elif task["name"] == "report_generation":
        incidents = [
            "At 14:30, traveler attempted to bring in undeclared alcohol. 3 bottles of whiskey found in luggage.",
            "Passport scanning revealed possible forgery. Security marking on page 4 inconsistent with issue date.",
            "Traveler John Smith appeared nervous, contradicted travel itinerary multiple times during questioning.",
            "Family of 4 had incomplete documentation for minor child. Missing parental consent form.",
            "Currency declaration form showed $5,000 but search revealed approximately $15,000 undeclared."
        ]
        
        for i in range(num_runs):
            input_text = task["input_template"].format(
                incident_details=incidents[i % len(incidents)]
            )
            
            inputs.append({
                "input": input_text,
                "expected_output": None  # Would need human evaluation for structure and completeness
            })
    
    else:
        # Default text generation inputs
        prompts = [
            "The border agent asked for",
            "When traveling internationally, always remember to",
            "At customs, the officer will typically",
            "The most important document for international travel is",
            "If stopped for additional screening, you should"
        ]
        
        for i in range(num_runs):
            input_text = task["input_template"].format(
                text=prompts[i % len(prompts)]
            )
            
            inputs.append({
                "input": input_text,
                "expected_output": None
            })
    
    return inputs

def _calculate_accuracy(output_text, expected_output):
    """Simple accuracy calculation - this could be more sophisticated"""
    output_lower = output_text.lower()
    
    # Check if expected output is contained in the model's response
    if expected_output.lower() in output_lower:
        return 1.0
    
    # Check for partial matches
    if any(word in output_lower for word in expected_output.lower().split()):
        return 0.5
        
    return 0.0

def _calculate_task_metrics(raw_data, metrics):
    """Calculate aggregate metrics from raw run data"""
    result_metrics = {}
    
    # Latency metrics
    latency_values = [run["simulated_time_ms"] for run in raw_data]
    result_metrics["avg_latency_ms"] = sum(latency_values) / len(latency_values)
    result_metrics["min_latency_ms"] = min(latency_values)
    result_metrics["max_latency_ms"] = max(latency_values)
    result_metrics["p95_latency_ms"] = np.percentile(latency_values, 95)
    
    # Accuracy metrics if available
    accuracy_values = [run.get("accuracy", None) for run in raw_data]
    if all(v is not None for v in accuracy_values):
        result_metrics["avg_accuracy"] = sum(accuracy_values) / len(accuracy_values)
    
    # Output length metrics
    output_lengths = [len(run["output"]) for run in raw_data]
    result_metrics["avg_output_length"] = sum(output_lengths) / len(output_lengths)
    
    # Custom metrics based on task needs
    for metric in metrics:
        if metric == "robustness" and "accuracy" in result_metrics:
            # Simplified robustness score based on accuracy consistency
            result_metrics["robustness"] = 1.0 - np.std(accuracy_values)
        
        elif metric == "quality" and "avg_output_length" in result_metrics:
            # Simplified quality heuristic based on output length
            result_metrics["estimated_quality"] = min(1.0, result_metrics["avg_output_length"] / 200)
    
    return result_metrics

def _calculate_overall_metrics(task_results):
    """Calculate overall benchmark metrics across all tasks"""
    overall_metrics = {}
    
    # Average latency across all tasks
    latencies = []
    for task_name, results in task_results.items():
        if "avg_latency_ms" in results["metrics"]:
            latencies.append(results["metrics"]["avg_latency_ms"])
    
    if latencies:
        overall_metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
        overall_metrics["max_task_latency_ms"] = max(latencies)
    
    # Average accuracy across tasks that have it
    accuracies = []
    for task_name, results in task_results.items():
        if "avg_accuracy" in results["metrics"]:
            accuracies.append(results["metrics"]["avg_accuracy"])
    
    if accuracies:
        overall_metrics["avg_accuracy"] = sum(accuracies) / len(accuracies)
    
    # Count tasks that meet latency requirements (<500ms)
    tasks_meeting_latency = sum(1 for l in latencies if l < 500)
    overall_metrics["latency_suitability"] = tasks_meeting_latency / len(latencies) if latencies else 0
    
    # Composite score (simplified)
    if "avg_accuracy" in overall_metrics and "latency_suitability" in overall_metrics:
        overall_metrics["edge_deployment_score"] = (
            overall_metrics["avg_accuracy"] * 0.6 + 
            overall_metrics["latency_suitability"] * 0.4
        )
    
    return overall_metrics
