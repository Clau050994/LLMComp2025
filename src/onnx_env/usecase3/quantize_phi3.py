import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

def quantize_phi3_simple():
    """Simple dynamic quantization for Phi3"""
    
    # Paths
    tokenizer_path = "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/phi3_onnx"
    model_fp32_path = "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/phi3_onnx/model.onnx"
    model_quant_path = "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/phi3_onnx/model_quantized.onnx"
    
    # Check if paths exist
    if not os.path.exists(model_fp32_path):
        print(f"❌ Model file not found: {model_fp32_path}")
        return
    
    print(f"📂 Input model: {model_fp32_path}")
    print(f"📂 Output model: {model_quant_path}")
    
    # Get original size
    original_size = os.path.getsize(model_fp32_path) / (1024*1024)  # MB
    print(f"📊 Original model size: {original_size:.1f} MB")
    
    try:
        print("🔄 Starting dynamic quantization...")
        
        # Simple dynamic quantization (minimal parameters)
        quantize_dynamic(
            model_input=model_fp32_path,
            model_output=model_quant_path,
            weight_type=QuantType.QInt8  # Only use supported parameters
        )
        
        print("✅ Dynamic quantization completed!")
        
        # Check new size
        if os.path.exists(model_quant_path):
            quantized_size = os.path.getsize(model_quant_path) / (1024*1024)  # MB
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            
            print(f"\n📊 Quantization Results:")
            print(f"Original size: {original_size:.1f} MB")
            print(f"Quantized size: {quantized_size:.1f} MB")
            print(f"Compression ratio: {compression_ratio:.1f}x")
            print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        # Test the quantized model
        print("\n🧪 Testing quantized model...")
        test_quantized_model(model_quant_path, tokenizer_path)
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()

def test_quantized_model(model_path, tokenizer_path):
    """Test the quantized model"""
    try:
        import onnxruntime as ort
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create session
        session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Using providers: {session.get_providers()}")
        
        # Test input
        test_text = "A traveler from Germany is entering the United States."
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        # Run inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        outputs = session.run(None, ort_inputs)
        predictions = np.argmax(outputs[0], axis=-1)
        
        print(f"✅ Inference test successful!")
        print(f"📊 Input shape: {inputs['input_ids'].shape}")
        print(f"📊 Output shape: {outputs[0].shape}")
        print(f"📊 Prediction: {predictions[0]}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    print("🚀 Starting Phi3 Dynamic Quantization")
    print("=" * 50)
    quantize_phi3_simple()
    print("=" * 50)
    print("🎉 Quantization process completed!")