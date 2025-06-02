# slm-sagemaker-eval
## Small Language Model (SLM) Evaluation on AWS SageMaker

This project evaluates the performance of 5–7 Small Language Models (SLMs) using AWS SageMaker. The goal is to benchmark these models for deployment on edge devices such as Raspberry Pi or Jetson Nano.

## 📌 Project Goals

- Train multiple open-source SLMs on SageMaker
- Benchmark latency, memory usage, accuracy, and model size
- Evaluate edge-deployability via quantization and ONNX export
- Prepare deployment pipelines for resource-constrained environments

📊 Evaluation Metrics
  - Latency
  - Accuracy
  - Model size (MB)
  - Inference time (ms)
  - Edge deployment feasibility

## 📦 Deployment Targets

  - Raspberry Pi (via AWS Greengrass)
  - Jetson Nano (via ONNX + TensorRT)
  - Simulated edge environments (Docker)

## 🧠 Models Under Evaluation
  - Phi-3-mini (128K context)
  - TinyLLaMA-1.1B
  - DistilBERT
  - ALBERT
  - MobileBERT
  - MobileLLaMA
  - Grok-3 mini-fast

## 👥 Contributors

| Name             | GitHub Username       | 
|------------------|------------------------|
| Claudia Saleem   | [@claudiasaleem](https://github.com/claudiasaleem) 
| Mostafa Elshabasy | [@Mosspheree](https://github.com/Mosspheree)       
       



