#!/usr/bin/env python3
"""
Exports a Hugging Face ALBERT classification model to ONNX format,
with dynamic axes and all tokenizer files copied.
"""

import os
import torch
import shutil
import numpy as np
import onnx
import onnxruntime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export_model_to_onnx(model_dir, onnx_dir, model_name="albert"):
    """Export a Hugging Face model to ONNX format with validation."""
    
    # Create output directory
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        # Set model to evaluation mode and disable gradients
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        return False

    # Prepare dummy input
    dummy_text = [
        "Traveler with French nationality, arriving from Brazil, last visited Morocco.",
        "What is the capital of France?",
        "Business visitor from USA with valid visa.",
        "Tourist from Germany traveling for vacation."
    ]

    print("Preparing dummy input...")
    inputs = tokenizer(
        dummy_text[0],  # Use first text for export
        return_tensors="pt",
        max_length=512,
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

    # Export to ONNX with dynamic axes
    print("Exporting model to ONNX...")
    try:
        torch.onnx.export(
            model,
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
            verbose=False,  # Changed to False to reduce output
        )

        print(f"✅ Exported ONNX model to {onnx_path}")
    except Exception as e:
        print(f"❌ Error exporting model to ONNX: {e}")
        return False

    # Verify the exported ONNX model
    print("\nVerifying ONNX model...")
    try:
        # Load and check the model 
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid.")
        print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")

        # Test inference with ONNX Runtime
        print("\nTesting inference with ONNX Runtime...")
        ort_session = onnxruntime.InferenceSession(onnx_path)

        # Prepare inputs for ONNX Runtime
        test_inputs = tokenizer(
            dummy_text[0],
            return_tensors="np",
            max_length=512,
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
                max_length=512,
                padding="max_length",
                truncation=True
            )
            torch_outputs = model(**torch_inputs)
            torch_logits = torch_outputs.logits.numpy()

        if np.allclose(ort_outputs[0], torch_logits, atol=1e-4):
            print("✅ ONNX output matches PyTorch output")
        else:
            print("⚠️  ONNX and PyTorch outputs differ slightly")
            print(f"Max difference: {np.max(np.abs(ort_outputs[0] - torch_logits))}")
    except Exception as e:
        print(f"❌ Error verifying ONNX model: {e}")
        return False

    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "config.json",  # Added config.json
        "merges.txt",
        "added_tokens.json" 
    ]

    print("\nCopying tokenizer files...")
    for file_name in tokenizer_files:
        src_file = os.path.join(model_dir, file_name)  # Fixed variable name
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
inputs = tokenizer(text, return_tensors="np", max_length=512, padding="max_length", truncation=True)

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
    model_dir = "/disk/diamond-scratch/cvaro009/data/mobilebert"
    onnx_dir = "/disk/diamond-scratch/cvaro009/data/onnx_models/mobilebert_onnx"
    
    # Export the model
    success = export_model_to_onnx(model_dir, onnx_dir, "mobilebert")

    if success:
        print("\n🎉 All done successfully!")
    else:
        print("\n❌ Export failed!")