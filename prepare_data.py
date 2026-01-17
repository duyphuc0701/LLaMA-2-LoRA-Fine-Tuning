import argparse
import os
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Format CodeAlpaca dataset for Llama 3 fine-tuning.")
    
    # Required argument: The output path
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True, 
        help="Directory where the processed dataset will be saved."
    )
    
    # Optional argument: Dataset name (defaults to CodeAlpaca)
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="sahil2801/CodeAlpaca-20k", 
        help="The Hugging Face dataset ID to load."
    )
    
    return parser.parse_args()

def format_llama3_code(sample):
    """
    Formats CodeAlpaca samples into Llama 3 chat template.
    """
    system_prompt = "You are an intelligent coding assistant that provides accurate and efficient code solutions."
    
    # Handle optional context/input
    user_message = sample['instruction']
    if sample.get('input'):
        user_message += f"\n\nInput:\n{sample['input']}"
        
    assistant_response = sample['output']

    # Llama 3 Format
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_response}<|eot_id|>"
    )
    return {"text": text}

def main():
    args = parse_arguments()
    
    print(f"Loading dataset: {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Original size: {len(dataset)} samples")
    
    # Apply formatting
    print("Formatting dataset...")
    processed_dataset = dataset.map(
        format_llama3_code, 
        remove_columns=dataset.column_names
    )
    
    # Verification
    print("\n=== SAMPLE 0 (Verification) ===")
    print(processed_dataset[0]['text'])
    print("===============================\n")

    # Save
    print(f"Saving processed dataset to: {args.save_path}")
    processed_dataset.save_to_disk(args.save_path)
    print("✅ Success! Dataset prepared.")

if __name__ == "__main__":
    main()