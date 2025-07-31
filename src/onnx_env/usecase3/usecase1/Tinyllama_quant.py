#!/usr/bin/env python3
"""
Export a TinyLLaMA model to ONNX and quantize it to INT8.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_and_quantize(model_dir, export_dir, opset=17):
    """
    Exports the model to ONNX and quantizes it.
    """
    os.makedirs(export_dir, exist_ok=True)

    onnx_fp32_path = os.path.join(export_dir, "tinyllama_fp32.onnx")
    onnx_quant_path = os.path.join(export_dir, "tinyllama_int8.onnx")

    print("🚀 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print("✅ Model and tokenizer loaded.")

    # Wrapper to avoid DynamicCache outputs
    class TinyLlamaForONNX(torch.nn.Module):
        def __init__(self, model):
            super(TinyLlamaForONNX, self).__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs.logits

    wrapped_model = TinyLlamaForONNX(model)

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
        wrapped_model,
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
    MODEL_DIR = "/aul/homes/melsh008/Generate/tinyllama_verification_finetuned"
    EXPORT_DIR = "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/tinyllama_onnx_quant"

    export_and_quantize(MODEL_DIR, EXPORT_DIR)
