import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed dataset")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ==============================================================================
    # 1. CONFIGURATION
    # ==============================================================================
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    NEW_MODEL_NAME = "Llama-3-8B-CodeAlpaca-Project"
    
    # ==============================================================================
    # 2. LOAD QUANTIZED MODEL (QLoRA)
    # ==============================================================================
    print(f"Loading {MODEL_NAME} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Disable cache to save memory during training (re-enable for inference)
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issues with fp16

    # ==============================================================================
    # 3. LOAD DATASET
    # ==============================================================================
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)
    
    # ==============================================================================
    # 4. LORA CONFIGURATION
    # ==============================================================================
    peft_config = LoraConfig(
        r=16,       # Rank (Higher = more parameters to train)
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    )

    # ==============================================================================
    # 5. TRAINING ARGUMENTS
    # ==============================================================================
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        packing=False,
    )

    # ==============================================================================
    # 6. START TRAINING
    # ==============================================================================
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()

    # ==============================================================================
    # 7. SAVE ADAPTERS
    # ==============================================================================
    print(f"Saving model to {args.output_dir}/{NEW_MODEL_NAME}")
    trainer.model.save_pretrained(f"{args.output_dir}/{NEW_MODEL_NAME}")
    tokenizer.save_pretrained(f"{args.output_dir}/{NEW_MODEL_NAME}")

if __name__ == "__main__":
    main()