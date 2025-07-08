#!/usr/bin/env python3
"""
Exports a Hugging Face TinyLlama classification model to ONNX format,
with dynamic axes and all tokenizer files copied.
"""

import os
import torch
import shutil
import numpy as np
import onnx
import onnxruntime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export_model_to_onnx(model_dir, onnx_dir, model_name="tinyllama"):
    """Export a Hugging Face TinyLlama model to ONNX format with validation."""
    
    # Create output directory
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    # Load model and tokenizer
    print("Loading TinyLlama model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Handle padding token for LLaMA models
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set pad_token to eos_token")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                print("Added new pad_token")
        
        model.eval()
        # Set model to evaluation mode and disable gradients
        for param in model.parameters():
            param.requires_grad = False
            
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        return False

    # Prepare dummy input - longer sequence for TinyLlama
    dummy_text = [
        "Traveler with French nationality, arriving from Brazil, last visited Morocco.",
        "Business visitor from USA with valid visa for conference attendance.",
        "Tourist from Germany traveling for vacation in summer.",
        "Student from Canada with study permit for university."
    ]

    print("Preparing dummy input...")
    # Use longer sequence length for TinyLlama
    max_length = 1024  # Updated to 1024 for TinyLlama
    inputs = tokenizer(
        dummy_text[0],  # Use first text for export
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    input_names = ["input_ids", "attention_mask"]
    
    # Verify input names
    missing_inputs = [name for name in input_names if name not in inputs]
    if missing_inputs:
        print(f"⚠️ Warning: Missing inputs in tokenizer output: {missing_inputs}")
        return False
    
    inputs_list = [inputs[name] for name in input_names]

    # Create a wrapper class to handle the cache issue
    class ONNXExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            # Force the model to not use cache for ONNX export
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,  # Disable cache
                return_dict=False  # Return tuple instead of dict
            )
            return outputs[0]  # Return only logits
    
    # Wrap the model
    wrapped_model = ONNXExportWrapper(model)
    wrapped_model.eval()

    # Export to ONNX with dynamic axes
    print("Exporting TinyLlama model to ONNX...")
    try:
        torch.onnx.export(
            wrapped_model,
            tuple(inputs_list),
            onnx_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        print(f"✅ Exported ONNX model to {onnx_path}")
    except Exception as e:
        print(f"❌ Error exporting model to ONNX: {e}")
        print("Trying alternative approach...")
        
        # Alternative approach: Try with older opset version
        try:
            torch.onnx.export(
                wrapped_model,
                tuple(inputs_list),
                onnx_path,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"}
                },
                opset_version=14,  # Older, more stable opset
                do_constant_folding=False,  # Disable constant folding
                export_params=True,
                verbose=False,
            )
            print(f"✅ Exported ONNX model to {onnx_path} (using opset 14)")
        except Exception as e2:
            print(f"❌ Alternative export also failed: {e2}")
            return False

    # Verify the exported ONNX model
    print("\nVerifying ONNX model...")
    try:
        # Check if file exists and get size
        if os.path.exists(onnx_path):
            model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"✅ ONNX model file created successfully ({model_size_mb:.1f} MB)")
        else:
            print("❌ ONNX model file not found")
            return False

        # Test inference with ONNX Runtime first (most important test)
        print("\nTesting inference with ONNX Runtime...")
        try:
            # Create ONNX Runtime session
            ort_session = onnxruntime.InferenceSession(onnx_path)
            
            # Get input/output info from session
            input_info = [(input.name, input.shape, input.type) for input in ort_session.get_inputs()]
            output_info = [(output.name, output.shape, output.type) for output in ort_session.get_outputs()]
            
            print(f"✅ ONNX Runtime session created successfully")
            print(f"Session inputs: {input_info}")
            print(f"Session outputs: {output_info}")

            # Prepare inputs for ONNX Runtime
            test_inputs = tokenizer(
                dummy_text[0],
                return_tensors="np",
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
            ort_inputs = {
                "input_ids": test_inputs["input_ids"].astype(np.int64),
                "attention_mask": test_inputs["attention_mask"].astype(np.int64)
            }

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"✅ ONNX Runtime inference successful")
            print(f"Output shape: {ort_outputs[0].shape}")

            # Compare with PyTorch output
            with torch.no_grad():
                torch_inputs = tokenizer(
                    dummy_text[0],
                    return_tensors="pt",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True
                )
                torch_outputs = wrapped_model(**torch_inputs)
                torch_logits = torch_outputs.numpy()

            if np.allclose(ort_outputs[0], torch_logits, atol=1e-3):
                print("✅ ONNX output matches PyTorch output")
            else:
                print("⚠️ ONNX and PyTorch outputs differ slightly")
                print(f"Max difference: {np.max(np.abs(ort_outputs[0] - torch_logits))}")
                
        except Exception as e:
            print(f"❌ Error testing ONNX Runtime inference: {e}")
            return False

        # Now try ONNX model validation (less critical)
        print("\nValidating ONNX model structure...")
        try:
            # Use path-based validation to avoid memory issues
            onnx.checker.check_model(onnx_path)
            print("✅ ONNX model structure is valid")
        except Exception as e:
            print(f"⚠️ ONNX model validation warning: {e}")
            print("✅ Model still functional (ONNX Runtime test passed)")
            
    except Exception as e:
        print(f"❌ Error in verification process: {e}")
        return False

    # Copy tokenizer files (LLaMA models have different tokenizer files)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "config.json",
        "tokenizer.model",  # SentencePiece model for LLaMA
        "added_tokens.json"
    ]

    print("\nCopying tokenizer files...")
    for file_name in tokenizer_files:
        src_file = os.path.join(model_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, onnx_dir)
            print(f"  ✔ Copied {file_name}")
        else:
            print(f"  ⚠️ {file_name} not found in model directory, skipping.")

    # Save tokenizer separately to ensure compatibility
    print("\nSaving tokenizer...")
    try:
        tokenizer.save_pretrained(onnx_dir)
        print(f"✅ Tokenizer saved to {onnx_dir}")
    except Exception as e:
        print(f"❌ Error saving tokenizer: {e}")

    # Create a simple usage example
    usage_example = f"""# Usage example for {model_name} ONNX model
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("{onnx_dir}")
ort_session = ort.InferenceSession("{onnx_path}")

# Prepare input
text = "Your input text here"
inputs = tokenizer(text, return_tensors="np", max_length={max_length}, padding="max_length", truncation=True)

# Run inference
ort_inputs = {{
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64)
}}

outputs = ort_session.run(None, ort_inputs)
logits = outputs[0]

# Get predictions
predictions = np.argmax(logits, axis=-1)
print(f"Predictions: {{predictions}}")
"""   
    
    try:
        with open(os.path.join(onnx_dir, "usage_example.py"), "w") as f:
            f.write(usage_example)
        print(f"📝 Usage example saved to {os.path.join(onnx_dir, 'usage_example.py')}")
    except Exception as e:
        print(f"⚠️ Error saving usage example: {e}")

    print(f"\n✅ Export complete! Files saved to {onnx_dir}")
    return True

if __name__ == "__main__":
    # Paths
    model_dir = "/disk/diamond-scratch/cvaro009/data/tinyllama"
    onnx_dir = "/disk/diamond-scratch/cvaro009/data/onnx_models/tinyllama_onnx"
    
    # Export the model
    success = export_model_to_onnx(model_dir, onnx_dir, "tinyllama")
    
    if success:
        print("\n🎉 TinyLlama export successful!")
    else:
        print("\n❌ TinyLlama export failed!")