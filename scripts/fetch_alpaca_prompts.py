import json
from pathlib import Path
from datasets import load_dataset

def main():
    print("Downloading massive high-quality prompt dataset (Alpaca Cleaned)...")
    try:
        # Alpaca cleaned has ~51k highly curated instruction-response pairs (~3-5 million tokens)
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
    except Exception as e:
        print(f"Error loading dataset from huggingface: {e}")
        print("Falling back to alternative dataset...")
        # Fallback to an alternative dataset if the first fails
        ds = load_dataset("HuggingFaceH4/no_robots", split="train")

    output_dir = Path("data/raw_prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "massive_custom_prompts.jsonl"
    
    print(f"Saving dataset to {output_file}...")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in ds:
            # Format instruction and response
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            output = row.get("output", "")
            
            prompt_text = ""
            if instruction:
                prompt_text += f"User: {instruction}\n"
            if input_text:
                prompt_text += f"Context: {input_text}\n"
            if output:
                prompt_text += f"Sturnus: {output}\n"
                
            if prompt_text:
                f.write(json.dumps({"text": prompt_text.strip()}) + "\n")
                count += 1

    print(f"Successfully generated {count:,} high-quality prompt pairs ready for ingestion!")

if __name__ == "__main__":
    main()
