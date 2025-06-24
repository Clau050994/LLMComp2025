"""
General local training module for the SLM evaluation framework.
Supports both classification and summarization-style models.
"""

import os
import logging
import sys
from typing import Dict, Any, Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

logger = logging.getLogger(__name__)


def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    dataset_csv: str,
    batch_size: int = 8,
    epochs: int = 3,
    output_path: Optional[str] = None
) -> str:
    logger.info(f"Starting local training for model {model_name}")
    return _train_locally(model_name, model_config, dataset_csv, batch_size, epochs, output_path)


def _train_locally(
    model_name: str,
    model_config: Dict[str, Any],
    dataset_csv: str,
    batch_size: int,
    epochs: int,
    output_path: Optional[str] = None
) -> str:
    try:
        logger.info(f"Loading dataset from {dataset_csv}")
        dataset = load_dataset("csv", data_files={"train": dataset_csv, "validation": dataset_csv})

        task_type = model_config.get("task_type", "classification")  # default to classification
        hf_model_id = model_config["hf_model_id"]
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

        if output_path is None:
            output_path = f"models/{model_name}/local/{epochs}epochs"
        os.makedirs(output_path, exist_ok=True)

        if task_type == "classification":
            logger.info("Detected task: classification")
            def preprocess(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length")

            tokenized = dataset.map(preprocess, batched=True)
            model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)

        elif task_type == "seq2seq":
            logger.info("Detected task: seq2seq")
            def preprocess(examples):
                inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
                targets = tokenizer(examples["label"], truncation=True, padding="max_length", max_length=128)
                inputs["labels"] = targets["input_ids"]
                return inputs

            tokenized = dataset.map(preprocess, batched=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id)

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            #evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(output_path, "logs"),
           # predict_with_generate=(task_type == "seq2seq")
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model) if task_type == "seq2seq" else None
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Evaluating model...")
        trainer.evaluate()

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"✅ Model and tokenizer saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        raise
