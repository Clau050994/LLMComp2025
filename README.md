# slm-sagemaker-eval
## Small Language Model (SLM) Evaluation on AWS SageMaker

This project evaluates the performance of 5–7 Small Language Models (SLMs) using AWS SageMaker. The goal is to benchmark these models for deployment on edge devices such as Raspberry Pi or Jetson Nano.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- AWS account with SageMaker access (for cloud training features)
- Git
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/claudiasaleem/slm-sagemaker-eval.git
   cd slm-sagemaker-eval
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Check system compatibility**

   Run a quick system check to verify your environment is properly set up:

   ```bash
   python run.py check
   ```

### Quick Start

1. **Train a model locally**

   ```bash
   python run.py train --model distilbert --dataset generic
   ```

2. **Train using AWS SageMaker**

   First ensure your AWS credentials are set up correctly, then:

   ```bash
   python run.py train --model distilbert --dataset generic --sagemaker
   ```

3. **Evaluate model performance**

   ```bash
   python run.py evaluate --model distilbert --device docker-sim
   ```

4. **Quantize model for edge deployment**

   ```bash
   python run.py quantize --model distilbert
   ```

5. **Deploy to edge device**

   ```bash
   python run.py deploy --model distilbert --device raspberry-pi --quantize
   ```

For more detailed usage, refer to the [Documentation.md](Documentation.md) file.

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

## 📁 Project Structure

```
slm-sagemaker-eval/
├── run.py                 # Simple CLI for running the framework
├── src/                   # Main source code
│   ├── models/            # Model definitions and registry
│   ├── training/          # Training functionality
│   ├── evaluation/        # Evaluation and benchmarking tools
│   ├── deployment/        # Deployment utilities
│   └── utils/             # Helper functions
├── config/                # Configuration files
├── data/                  # Data directory
└── notebooks/             # Analysis notebooks
```

## ⚙️ Platform Compatibility

This framework is designed to work across different operating systems:

- **Linux**: Fully supported and recommended for production use
- **macOS**: Fully supported for development and testing
- **Windows**: Supported with some additional setup requirements:
  - Install Git Bash or WSL (Windows Subsystem for Linux) for better compatibility
  - For CUDA support, install the appropriate NVIDIA drivers and CUDA toolkit
  - Some packages may require specific Windows versions

Run `python run.py check` to verify compatibility with your system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

| Name             | GitHub Username       | 
|------------------|------------------------|
| Claudia Saleem   | [@claudiasaleem](https://github.com/claudiasaleem) 
| Mostafa Elshabasy | [@Mosspheree](https://github.com/Mosspheree)       
       



