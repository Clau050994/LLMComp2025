"""
Conversion utilitie    Convert a model to ONNX format for better interoperability.
    
    Args:
        model_name (str): Name of the model to convert
        model_config (Dict[str, Any]): Model configuration
        input_sample (Optional[Tuple[torch.Tensor, ...]]): Sample input tensor for tracingSLMs to different formats.
Primarily supports conversion to ONNX format for better optimization and compatibility.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

def convert_to_onnx(
    model_name: str,
    model_config: Dict[str, Any],
    input_sample: Optional[Tuple[torch.Tensor, ...]] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    opset_version: int = 14,
    simplify: bool = True
) -> str:
    """
    Convert a model to ONNX format for edge deployment.
    
    Args:
        model_name (str): Name of the model to convert
        model_config (Dict[str, Any]): Model configuration
        input_sample (Tuple[torch.Tensor, ...], optional): Sample input for tracing
        model_path (str, optional): Path to the model file
        output_path (str, optional): Path to save the ONNX model
        opset_version (int): ONNX opset version
        simplify (bool): Whether to simplify the ONNX model
        
    Returns:
        str: Path to the ONNX model
    """
    if not model_config["onnx_supported"]:
        logger.warning(f"Model {model_name} may not be fully supported for ONNX export")
    
    logger.info(f"Converting model {model_name} to ONNX format")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Set default paths if not provided
        if model_path is None:
            model_id = model_config["hf_model_id"]
        else:
            model_id = model_path
            
        if output_path is None:
            output_path = f"models/{model_name}/onnx"
            os.makedirs(output_path, exist_ok=True)
            
        onnx_path = os.path.join(output_path, "model.onnx")
        
        # Load the model and tokenizer
        logger.info(f"Loading model from {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Create sample input if not provided
        if input_sample is None:
            logger.info("Creating sample input for ONNX tracing")
            sample_text = "This is a sample input for ONNX conversion."
            inputs = tokenizer(sample_text, return_tensors="pt")
            input_sample = (
                inputs["input_ids"],
                inputs.get("attention_mask", None),
                inputs.get("token_type_ids", None)
            )
            
            # Filter out None values
            input_sample = tuple(x for x in input_sample if x is not None)
        
        # Export to ONNX
        logger.info(f"Exporting model to ONNX (opset_version={opset_version})")
        
        # Set the model to evaluation mode
        model.eval()
        
        # Export using torch.onnx.export
        torch.onnx.export(
            model,                      # model being run
            input_sample,               # model input (or a tuple for multiple inputs)
            onnx_path,                  # where to save the model
            export_params=True,         # store the trained parameter weights inside the model file
            opset_version=opset_version,# the ONNX version to export the model to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=['input_ids'],  # the model's input names
            output_names=['output'],    # the model's output names
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            }
        )
        
        # Simplify the ONNX model if requested
        if simplify:
            logger.info("Simplifying ONNX model")
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                # Load the model
                onnx_model = onnx.load(onnx_path)
                
                # Simplify
                simplified_model, check = onnx_simplify(onnx_model)
                
                if check:
                    # Save the simplified model
                    onnx.save(simplified_model, onnx_path)
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX model could not be simplified")
            except ImportError:
                logger.warning("onnxsim not installed, skipping model simplification")
                
        # Save the tokenizer alongside the model
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"Model converted and saved to {onnx_path}")
        
        # Create a config file to document the conversion
        with open(os.path.join(output_path, "conversion_config.json"), "w") as f:
            f.write(f'{{"model_name": "{model_name}", "opset_version": {opset_version}, "simplified": {simplify}}}')
        
        return onnx_path
        
    except Exception as e:
        logger.error(f"Error in ONNX conversion: {str(e)}", exc_info=True)
        raise
