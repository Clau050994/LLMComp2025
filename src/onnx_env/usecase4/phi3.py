#!/usr/bin/env python3
"""
Exports a Hugging Face Phi-3 QLoRA classification model to ONNX format.
Supports 4-bit quantized LoRA adapters and dynamic axes export.
"""

import os
import torch
import shutil
import numpy as np
import onnx
import onnxruntime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, PeftConfig


def export_phi3_qlora_to_onnx(model_dir, onnx_output_dir, model_name="phi3"):
    os.makedirs(onnx_output_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_output_dir, "model.onnx")

    print("🔄 Loading tokenizer and QLoRA model ...")
    try:
        # 1️ Load PEFT config
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        print(f"Base model: {peft_cfg.base_model_name_or_path}")

        # 2️ Load base model config and set correct number of labels
        cfg = AutoConfig.from_pretrained(peft_cfg.base_model_name_or_path)
        cfg.num_labels = 3

        # 3️ Load base model WITHOUT quantization (for ONNX export)
        print("Loading base model without quantization...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            config=cfg,
            torch_dtype=torch.float32,  # Use float32 for ONNX compatibility
            device_map="cpu",  # Load on CPU to avoid GPU memory issues
            ignore_mismatched_sizes=True
        )

        # 4️ Load and merge LoRA adapter
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_dir)
        
        print("Merging LoRA adapter...")
        model = model.merge_and_unload()
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # 5️ Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model and tokenizer loaded successfully")

    except Exception as e:
        print(f"❌ Error loading or merging QLoRA model: {e}")
        return False

    # Dummy input with max_length=1024
    print("Preparing dummy input...")
    dummy_text = "Business traveler from Argentina entering the U.S. for a 10-day trip."
    max_length = 1024  # Updated to 1024
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt", 
        max_length=max_length, 
        padding="max_length", 
        truncation=True
    )
    input_names = ["input_ids", "attention_mask"]
    inputs_list = [inputs[name] for name in input_names]

    # Create a wrapper to handle potential issues
    class ONNXExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False
            )
            return outputs[0]  # Return only logits
    
    wrapped_model = ONNXExportWrapper(model)
    wrapped_model.eval()

    print("🧠 Exporting to ONNX...")
    try:
        torch.onnx.export(
            wrapped_model,
            tuple(inputs_list),
            onnx_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"}
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False
        )
        print(f"✅ Exported ONNX model to {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("Trying with opset version 14...")
        try:
            torch.onnx.export(
                wrapped_model,
                tuple(inputs_list),
                onnx_path,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"},
                    "logits": {0: "batch_size"}
                },
                opset_version=14,
                do_constant_folding=False,
                verbose=False
            )
            print(f"✅ Exported ONNX model to {onnx_path} (opset 14)")
        except Exception as e2:
            print(f"❌ ONNX export failed with opset 14: {e2}")
            return False

    # Validate ONNX model
    print("✅ Validating ONNX model...")
    try:
        # Check file size first
        if os.path.exists(onnx_path):
            model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"ONNX model size: {model_size_mb:.1f} MB")
            
            # Test with ONNX Runtime
            ort_session = onnxruntime.InferenceSession(onnx_path)
            
            # Get model info
            input_info = [(inp.name, inp.shape, inp.type) for inp in ort_session.get_inputs()]
            output_info = [(out.name, out.shape, out.type) for out in ort_session.get_outputs()]
            
            print(f"Inputs: {input_info}")
            print(f"Outputs: {output_info}")

            # Test inference with max_length=1024
            np_inputs = tokenizer(
                dummy_text, 
                return_tensors="np", 
                max_length=max_length, 
                padding="max_length", 
                truncation=True
            )
            ort_inputs = {k: np_inputs[k].astype(np.int64) for k in input_names}
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"✅ ONNX Runtime output shape: {ort_outputs[0].shape}")
            
            # Compare with PyTorch output
            with torch.no_grad():
                torch_inputs = tokenizer(
                    dummy_text,
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
                
        else:
            print("❌ ONNX model file not found")
            return False
            
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False

    print("📂 Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer_config.json", 
        "tokenizer.json", 
        "special_tokens_map.json", 
        "vocab.txt", 
        "config.json"
    ]
    
    for f in tokenizer_files:
        src = os.path.join(model_dir, f)
        if os.path.exists(src):
            shutil.copy(src, onnx_output_dir)
            print(f"✔ Copied {f}")
        else:
            print(f"⚠️ {f} not found, skipping")

    try:
        tokenizer.save_pretrained(onnx_output_dir)
        print("✅ Tokenizer saved successfully")
    except Exception as e:
        print(f"⚠️ Failed to save tokenizer: {e}")

    # Example script with max_length=1024
    usage_example = f"""# Usage example for {model_name} ONNX model
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("{onnx_output_dir}")
ort_session = ort.InferenceSession("{onnx_path}")

text = "Your input text here"
inputs = tokenizer(text, return_tensors="np", max_length={max_length}, padding="max_length", truncation=True)
ort_inputs = {{
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64)
}}
outputs = ort_session.run(None, ort_inputs)
logits = outputs[0]
predictions = np.argmax(logits, axis=-1)
print(f"Predictions: {{predictions}}")
"""
    with open(os.path.join(onnx_output_dir, "usage_example.py"), "w") as f:
        f.write(usage_example)

    print(f"\n✅ Export complete! Files saved to {onnx_output_dir}")
    return True


if __name__ == "__main__":
    model_dir = "/disk/diamond-scratch/cvaro009/data/phi3_risk_classification_qlora"
    onnx_dir = "/disk/diamond-scratch/cvaro009/data/onnx_models/phi3_onnx"
    
    success = export_phi3_qlora_to_onnx(model_dir, onnx_dir, "phi3")
    if success:
        print("\n🎉 All done successfully!")
    else:
        print("\n❌ Export failed!")