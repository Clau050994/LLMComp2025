# LLMComp2025: Small Language Model Optimization & Evaluation Framework

A comprehensive framework for evaluating, optimizing, and benchmarking Small Language Models (SLMs) with a focus on edge deployment, ONNX optimization, and robustness testing. This project provides extensive evaluation of 6 state-of-the-art small language models across multiple metrics including accuracy, inference time, memory usage, model size, and robustness.

## 🎯 Key Features

- **Complete ONNX Pipeline**: Export, optimize and evaluate models in ONNX format for production deployment
- **Comprehensive Benchmarking**: Multi-dimensional evaluation including clean accuracy, robustness testing, and performance metrics
- **Energy Efficiency Analysis**: Detailed energy consumption analysis across different hardware platforms (GPU, CPU, laptop, mobile)
- **Edge Deployment Ready**: Optimized for deployment on resource-constrained devices
- **Robust Evaluation**: Includes noise injection and adversarial testing to measure model robustness
- **Automated Model Comparison**: Side-by-side comparison of all models with detailed performance summaries

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (for larger models)
- 50GB+ disk space

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd LLMComp2025
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or venv\Scripts\activate on Windows
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Evaluate a pre-trained model**

   ```bash
   python test_models.py
   ```

2. **Run ONNX evaluation on all models**

   ```bash
   python src/evaluation/usecase3/final_comparison.py
   ```

3. **Generate visualization plots**

   ```bash
   python src/evaluation/usecase3/plots_ONNX/general_plots_onnx.py
   ```

4. **Export a model to ONNX**

   ```bash
   python src/onnx_env/usecase3/distilbert.py  # Replace with desired model
   ```

## 📊 Evaluation Results

Our comprehensive evaluation reveals the following performance characteristics:

| Model | Clean Accuracy | Robust Accuracy | Inference Time (ms) | Model Size (MB) | Energy/Query (CPU) |
|-------|----------------|-----------------|--------------------|-----------------|--------------------|
| **TinyLLaMA** | 93.27% | 89.11% | 15.40 | 4.49 | 0.616J |
| **Phi-3-mini** | 92.17% | 84.62% | 69.80 | 2213.84 | 2.792J |
| **MobileBERT** | 91.87% | 78.35% | 1.67 | 95.36 | 0.067J |
| **DistilBERT** | 91.70% | 79.61% | 1.13 | 256.43 | 0.045J |
| **ALBERT** | 91.45% | 78.05% | 2.21 | 47.11 | 0.088J |
| **MobileLLaMA** | 91.38% | 80.50% | 0.55 | 169.31 | 0.022J |

### Key Findings:
- **Best Overall Performance**: TinyLLaMA achieves the highest accuracy with good efficiency
- **Fastest Inference**: MobileLLaMA provides sub-millisecond inference times
- **Most Robust**: TinyLLaMA maintains 89% accuracy under adversarial conditions
- **Most Energy Efficient**: MobileLLaMA consumes the least energy per query

## 🧠 Supported Models

- **Phi-3-mini**: Microsoft's efficient 3.8B parameter model with 128K context
- **TinyLLaMA-1.1B**: Compact yet capable language model optimized for efficiency
- **DistilBERT**: Knowledge-distilled BERT with 66M parameters
- **ALBERT**: Parameter-efficient architecture with 12M parameters
- **MobileBERT**: Optimized BERT variant for mobile deployment
- **MobileLLaMA**: Mobile-optimized LLaMA variant for edge devices

## 🏗️ Project Structure

```
LLMComp2025/
├── README.md                    # This file
├── Documentation.md             # Detailed technical documentation
├── requirements.txt             # Python dependencies
├── run.py                      # Command-line interface
├── test_models.py              # Quick model testing script
├── config/                     # Configuration files
│   ├── config.json            # Model configurations
│   └── evaluation_config.py   # Evaluation settings
├── src/                       # Source code
│   ├── main.py               # Main entry point
│   ├── training/             # Training scripts for use cases
│   ├── evaluation/           # Evaluation and benchmarking
│   ├── onnx_env/            # ONNX export and optimization
│   └── utils/               # Utility functions
├── data/                      # Dataset storage
│   ├── raw/                  # Original datasets
│   └── processed/            # Preprocessed datasets
├── Data_scripts/              # Data preprocessing scripts
├── onnx_results_usecase3*/    # ONNX evaluation results
└── results_baseline_comparison/ # Baseline comparison results
```

## 📈 Evaluation Framework

### Core Evaluation Metrics
- **Accuracy**: Classification performance on clean test data
- **Robustness**: Performance degradation under adversarial conditions (noise injection)
- **Inference Time**: Average time per prediction in milliseconds
- **Model Size**: Disk storage requirements in MB
- **Memory Usage**: Peak RAM consumption during inference
- **Energy Efficiency**: Power consumption per query across different hardware

### ONNX Optimization Pipeline
1. **Model Export**: Convert PyTorch models to ONNX format
2. **Optimization**: Graph optimization and quantization
3. **Validation**: Ensure output consistency between frameworks
4. **Benchmarking**: Comprehensive performance evaluation

### Robustness Testing
- Character-level noise injection (typos, substitutions)
- Input perturbation analysis
- Adversarial robustness evaluation
- Cross-dataset generalization testing

## 🔧 Usage Examples

### Evaluate All Models
```bash
# Run complete ONNX evaluation pipeline
python src/evaluation/usecase3/final_comparison.py

# Generate comparative visualization
python src/evaluation/usecase3/plots_ONNX/general_plots_onnx.py
```

### Export Individual Models to ONNX
```bash
# Export DistilBERT
python src/onnx_env/usecase3/distilbert.py

# Export ALBERT
python src/onnx_env/usecase3/albert.py

# Export TinyLLaMA
python src/onnx_env/usecase3/tinyllama.py
```

### Model-Specific Evaluation
```bash
# Evaluate ALBERT with detailed metrics
python src/evaluation/usecase3/albert_eval.py

# Evaluate Phi-3 with robustness testing
python src/evaluation/usecase3/eval_phi3.py
```

### Quick Interactive Testing
```bash
# Test any model interactively
python test_models.py
# Enter text when prompted to get real-time predictions
```

## 📊 Results and Insights

### Performance vs Efficiency Trade-offs
- **High Accuracy + Fast**: TinyLLaMA offers the best balance
- **Ultra-Fast Inference**: MobileLLaMA for real-time applications
- **Memory Efficient**: ALBERT for resource-constrained environments
- **Large Context**: Phi-3-mini for complex reasoning tasks

### Energy Consumption Analysis
Our energy analysis across different hardware platforms shows:
- **Mobile Devices**: MobileLLaMA consumes 10x less energy than Phi-3
- **Laptops**: DistilBERT provides best energy efficiency for mid-range performance
- **Servers**: All models show excellent GPU utilization

### Robustness Findings
- **Most Robust**: TinyLLaMA maintains 95.7% of original performance under noise
- **Least Robust**: ALBERT shows 85.3% retention under adversarial conditions
- **Noise Types**: Character substitution has minimal impact vs. word-level corruption

## 🛠️ Advanced Features

### Custom Model Integration
```python
# Add your own model to the evaluation framework
from src.models import register_model

@register_model("your_model")
def load_your_model():
    # Model loading logic
    return model, tokenizer
```

### Custom Evaluation Metrics
```python
# Extend evaluation with custom metrics
from src.evaluation import add_custom_metric

@add_custom_metric
def custom_efficiency_score(predictions, ground_truth, inference_times):
    # Your metric calculation
    return score
```

### Hardware-Specific Optimization
The framework supports optimization for different target devices:
- **Mobile/Edge**: Quantization + pruning for minimal footprint
- **Desktop**: Balanced optimization for speed/accuracy
- **Server**: Multi-batch optimization for throughput

## 📋 Requirements

### System Requirements
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 50GB+ for models and datasets
- **GPU**: CUDA-compatible (optional but recommended)
- **CPU**: Multi-core processor (4+ cores recommended)

### Software Dependencies
See `requirements.txt` for complete list. Key dependencies:
- PyTorch >= 2.0.0
- Transformers >= 4.34.0
- ONNX >= 1.14.0
- ONNXRuntime >= 1.15.0
- scikit-learn >= 1.2.0
- pandas >= 2.0.0

## 🚀 Reproducibility

### Running the Complete Evaluation
```bash
# 1. Set up environment
source venv/bin/activate

# 2. Run full evaluation pipeline
python src/evaluation/usecase3/final_comparison.py

# 3. Generate all plots and visualizations
python src/evaluation/usecase3/plots_ONNX/general_plots_onnx.py

# 4. Review results
ls onnx_results_usecase3/
```

### Expected Outputs
- Model comparison CSV files
- Confusion matrices for each model
- Performance visualization plots
- Energy consumption analysis
- Individual model evaluation reports

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

## 🏆 Acknowledgments

This work was conducted as part of LLMComp2025, focusing on advancing the state of small language model optimization for practical deployment scenarios. Special thanks to the open-source community for providing the foundational models and tools that made this research possible.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{llmcomp2025,
  title={LLMComp2025: Small Language Model Optimization \& Evaluation Framework},
  author={Saleem, Claudia and Elshabasy, Mostafa},
  year={2025},
  howpublished={\url{https://github.com/your-repo/LLMComp2025}}
}
```