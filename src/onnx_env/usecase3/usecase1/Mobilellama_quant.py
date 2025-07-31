#!/usr/bin/env python3
"""
Exports a Hugging Face MobileLLaMA classification model to ONNX format,
avoiding DynamicCache issues by wrapping the forward pass.
"""

import os
import torch
import shutil
import numpy as np
import onnx
import onnxruntime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export_model_to_onnx(model_dir, onnx_dir, model_name="mobilellama"):
    """Export MobileLLaMA to ONNX, safely avoiding DynamicCache."""
    
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        return False

    # ✅ Wrapped model to output only logits
    class ModelForONNX(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs.logits

    model = ModelForONNX(base_model)

    # Prepare dummy input
    dummy_text = "Business visitor from USA with valid visa."
    print("Preparing dummy input...")
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True
    )
    input_names = ["input_ids", "attention_mask"]
    inputs_list = [inputs[name] for name in input_names]

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
            verbose=False,
        )
        print(f"✅ Exported ONNX model to {onnx_path}")
    except Exception as e:
        print(f"❌ Error exporting model to ONNX: {e}")
        return False

    print("\nVerifying ONNX model...")
    try:
        onnx.checker.check_model(onnx_path)
        print("✅ ONNX model is valid.")
        ort_session = onnxruntime.InferenceSession(onnx_path)

        test_inputs = tokenizer(
            dummy_text,
            return_tensors="np",
            max_length=256,
            padding="max_length",
            truncation=True
        )
        ort_inputs = {
            "input_ids": test_inputs["input_ids"].astype(np.int64),
            "attention_mask": test_inputs["attention_mask"].astype(np.int64)
        }

        ort_outputs = ort_session.run(None, ort_inputs)
        print("✅ ONNX Runtime inference successful")
        print(f"Output shape: {ort_outputs[0].shape}")

        with torch.no_grad():
            torch_outputs = model(
                torch.tensor(ort_inputs["input_ids"]),
                torch.tensor(ort_inputs["attention_mask"])
            )
            torch_logits = torch_outputs.numpy()

        if np.allclose(ort_outputs[0], torch_logits, atol=1e-4):
            print("✅ ONNX output matches PyTorch output")
        else:
            print("⚠ ONNX and PyTorch outputs differ slightly")
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
        "config.json",
        "merges.txt",
        "added_tokens.json"
    ]

    print("\nCopying tokenizer files...")
    for file_name in tokenizer_files:
        src = os.path.join(model_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, onnx_dir)
            print(f"  ✔ Copied {file_name}")
        else:
            print(f"  ⚠ {file_name} not found, skipping.")

    print("\nSaving tokenizer...")
    try:
        tokenizer.save_pretrained(onnx_dir)
        print(f"✅ Tokenizer saved to {onnx_dir}")
    except Exception as e:
        print(f"❌ Error saving tokenizer: {e}")

    print(f"\n✅ Export complete! Files saved to {onnx_dir}")
    return True

if __name__ == "__main__":
    model_dir = "/aul/homes/melsh008/Generate/mobilellama_verification_finetuned"
    onnx_dir = "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/mobilellama_export_onnx"

    success = export_model_to_onnx(model_dir, onnx_dir, "mobilellama")

    if success:
        print("\n🎉 All done successfully!")
    else:
        print("\n❌ Export failed!")
