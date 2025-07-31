#!/usr/bin/env python3
"""
Export a MobileBERT model to ONNX and quantize it to INT8.
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

    onnx_fp32_path = os.path.join(export_dir, "mobilebert_fp32.onnx")
    onnx_quant_path = os.path.join(export_dir, "mobilebert_int8.onnx")

    print("🚀 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print("✅ Model and tokenizer loaded.")

    # === Wrapper to return only logits
    class LogitsWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    wrapped = LogitsWrapper(model)

    # === Prepare dummy input
    dummy_inputs = tokenizer(
        "This is a test input for ONNX export.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    print("💾 Exporting FP32 ONNX...")
    torch.onnx.export(
        wrapped,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_fp32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=opset,
        do_constant_folding=True
    )
    print(f"✅ FP32 ONNX saved to {onnx_fp32_path}")

    print("⚙️ Quantizing to INT8...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_quant_path,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized INT8 ONNX saved to {onnx_quant_path}")

    print("💾 Saving tokenizer...")
    tokenizer.save_pretrained(export_dir)
    print(f"✅ Tokenizer saved to {export_dir}")

if __name__ == "__main__":
    MODEL_DIR = "/a/buffalo.cs.fiu.edu./disk/jccl-002/homes/melsh008/Generate/mobilebert_verification_finetuned/checkpoint-30005"
    EXPORT_DIR = "/disk/diamond-scratch/melsh008/mobilebert_export_onnx"

    export_and_quantize(MODEL_DIR, EXPORT_DIR)
