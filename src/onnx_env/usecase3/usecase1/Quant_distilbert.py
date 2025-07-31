#!/usr/bin/env python3
"""
Export a DistilBERT model to ONNX and quantize it to INT8.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_and_quantize(model_dir, export_dir, opset=14):
    """
    Exports the model to ONNX and quantizes it.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"❌ The model directory does not exist: {model_dir}")

    os.makedirs(export_dir, exist_ok=True)

    onnx_fp32_path = os.path.join(export_dir, "distilbert_fp32.onnx")
    onnx_quant_path = os.path.join(export_dir, "distilbert_int8.onnx")

    print("🚀 Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"⚠️ Tokenizer not found in checkpoint: {e}")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print("✅ Model and tokenizer loaded.")

    # Prepare dummy input
    dummy_inputs = tokenizer(
        "This is a test input for ONNX export.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    print("💾 Exporting FP32 ONNX model...")
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_fp32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=opset,
        do_constant_folding=True
    )
    print(f"✅ FP32 ONNX model saved to {onnx_fp32_path}")

    print("⚙️ Quantizing to INT8...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_quant_path,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized INT8 ONNX model saved to {onnx_quant_path}")

    print("💾 Saving tokenizer...")
    tokenizer.save_pretrained(export_dir)
    print(f"✅ Tokenizer saved to {export_dir}")

if __name__ == "__main__":
    MODEL_DIR = "/a/buffalo.cs.fiu.edu./disk/jccl-002/homes/melsh008/Generate/distilbert_verification_finetuned/checkpoint-42007"
    EXPORT_DIR = "/disk/diamond-scratch/melsh008/distilbert_export_onnx"

    export_and_quantize(MODEL_DIR, EXPORT_DIR)
