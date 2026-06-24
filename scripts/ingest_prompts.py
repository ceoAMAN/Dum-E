import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path so we can import configs and tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import get_tokenizer
import configs

def process_file(file_path: Path) -> list:
    """Reads a file and returns a list of text samples."""
    samples = []
    try:
        if file_path.suffix.lower() == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            samples.append(data["text"])
                        elif "prompt" in data:
                            samples.append(data["prompt"])
                        elif "instruction" in data:
                            samples.append(data.get("instruction", "") + "\n" + data.get("output", ""))
                        else:
                            # Fallback: dump json string
                            samples.append(json.dumps(data))
                    except:
                        pass
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            samples.append(item["text"])
                        elif isinstance(item, str):
                            samples.append(item)
                elif isinstance(data, dict):
                    if "text" in data:
                        samples.append(data["text"])
        else:
            # For txt, md, read the whole file as one prompt
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    samples.append(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return samples

def main():
    parser = argparse.ArgumentParser(description="Ingest raw prompts into Sturnus custom stream.")
    parser.add_argument("--input", type=str, default="data/raw_prompts", help="Directory containing raw prompt files (.txt, .md, .json, .jsonl)")
    parser.add_argument("--output", type=str, default="data/custom_prompts.jsonl", help="Output file for the streaming dataset")
    parser.add_argument("--test", action="store_true", help="Run in test mode (dummy data)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)
    
    # Ensure data dir exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.test and not input_dir.exists():
        print(f"Creating dummy test file in {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        with open(input_dir / "test_prompt.txt", "w", encoding="utf-8") as f:
            f.write("This is a massive test prompt to simulate a local custom prompt ingest.\n" * 10)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print(f"Please create '{input_dir}' and drop your 2 million tokens of prompt files (.txt, .md, .jsonl) inside it.")
        return

    print("Loading MLX Tokenizer to compute exact token ingestion volume...")
    try:
        tokenizer = get_tokenizer(configs.EXPERT_MODEL_ID)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer. We will use a fast character-based approximation. Error: {e}")
        tokenizer = None

    print(f"\nScanning {input_dir} for prompt files...")
    
    total_tokens = 0
    total_files = 0
    total_samples = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.startswith('.'): continue
                file_path = Path(root) / file
                
                samples = process_file(file_path)
                if not samples:
                    continue
                    
                total_files += 1
                
                for text in samples:
                    # Token count
                    if tokenizer:
                        # encode gives list of ids
                        token_count = len(tokenizer.encode(text))
                    else:
                        token_count = len(text) // 4
                    
                    total_tokens += token_count
                    total_samples += 1
                    
                    # Write as JSONL
                    row = {"text": text}
                    out_f.write(json.dumps(row) + "\n")
                    
    print("\n" + "="*50)
    print("INGESTION COMPLETE")
    print("="*50)
    print(f"Files Processed: {total_files:,}")
    print(f"Total Prompts:   {total_samples:,}")
    print(f"Total Tokens:    {total_tokens:,}")
    print("="*50)
    
    if total_tokens < 2_000_000:
        print(f"\nNote: You have ingested {total_tokens:,} tokens.")
        print("To hit your 2+ million token goal, please drop more files into 'data/raw_prompts/' and run this script again.")
    else:
        print(f"\nSUCCESS: You hit your goal of 2+ million tokens ({total_tokens:,})!")
        
    print(f"\nYour stream is ready at: {output_file}")
    print("Sturnus will now automatically prioritize these custom prompts during inference background dead-time.")

if __name__ == "__main__":
    main()
