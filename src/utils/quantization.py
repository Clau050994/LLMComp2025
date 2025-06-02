"""
Quantization utilities for SLMs.
"""

import os
import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

def quantize_model(
    model_name: str,
    model_config: Dict[str, Any],
    bits: int = 4,
    model_path: str = None,
    output_path: Optional[str] = None
) -> str:
    """
    Quantize a model to reduce its size and memory footprint.
    
    Args:
        model_name (str): Name of the model to quantize
        model_config (Dict[str, Any]): Model configuration
        bits (int): Quantization bits (4 or 8)
        model_path (str, optional): Path to the model file
        output_path (str, optional): Path to save the quantized model
        
    Returns:
        str: Path to the quantized model
    """
    if not model_config["quantization_supported"]:
        logger.warning(f"Model {model_name} does not support quantization")
        return None
    
    logger.info(f"Quantizing model {model_name} to {bits} bits")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Set default paths if not provided
        if model_path is None:
            model_id = model_config["hf_model_id"]
        else:
            model_id = model_path
            
        if output_path is None:
            output_path = f"models/{model_name}/quantized_{bits}bit"
            os.makedirs(output_path, exist_ok=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Quantize the model
        if bits == 4:
            logger.info("Using 4-bit quantization")
            
            try:
                from transformers import BitsAndBytesConfig
                
                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load model with quantization config
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                
                # Save the model (this will save in safetensors format)
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                
                # Create a config indicating this is quantized
                with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
                    f.write('{"bits": 4, "quant_method": "bitsandbytes"}')
                    
                logger.info(f"4-bit quantized model saved to {output_path}")
                
            except ImportError:
                logger.warning("BitsAndBytes not available for 4-bit quantization.")
                logger.info("Falling back to 8-bit quantization")
                return quantize_model(model_name, model_config, bits=8, model_path=model_path, output_path=output_path)
                
        elif bits == 8:
            logger.info("Using 8-bit quantization")
            
            try:
                import torch.quantization
                
                # Load the model first
                model = AutoModelForCausalLM.from_pretrained(model_id)
                
                # Quantize the model to 8-bit
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
                # Save the quantized model
                quantized_model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                
                # Create a config indicating this is quantized
                with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
                    f.write('{"bits": 8, "quant_method": "pytorch_dynamic"}')
                    
                logger.info(f"8-bit quantized model saved to {output_path}")
                
            except Exception as e:
                logger.error(f"Error in 8-bit quantization: {str(e)}", exc_info=True)
                raise
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}. Use 4 or 8.")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error in model quantization: {str(e)}", exc_info=True)
        raise
