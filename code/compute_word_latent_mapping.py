import os
import sys
import argparse
import torch
from transformers import GPT2Tokenizer

# Ensure code directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from demonstrate import compute_word_latent_mapping_on_dataset
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE

def main():
    parser = argparse.ArgumentParser(description="Compute word-to-latent mapping for EnhancedGPT2VQVAE on a dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.pt or .ckpt)')
    parser.add_argument('--data_dir', type=str, required=False, help='Path to the data directory (optional, will use GSM8K default if not provided)')
    parser.add_argument('--split_name', type=str, choices=['train', 'test'], required=True, help="Which split to analyze: 'train' or 'test'")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data loading')
    args = parser.parse_args()

    print(f"Loading EnhancedGPT2VQVAE from checkpoint: {args.checkpoint_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EnhancedGPT2VQVAE.from_checkpoint(args.checkpoint_path, device=device)
    model.eval()
    print("Model loaded.")

    num_thoughts = getattr(model, 'num_thoughts', None)
    if num_thoughts is None:
        raise AttributeError("Loaded model does not have 'num_thoughts' attribute.")
    print(f"Detected num_thoughts from model: {num_thoughts}")

    # Set data_dir if not provided
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = f"data/GSM8K/128_128/batch_{num_thoughts}"
        print(f"No --data_dir provided. Using default: {data_dir}")
    else:
        print(f"Using provided data_dir: {data_dir}")

    print("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    print(f"Running word-to-latent mapping analysis on {args.split_name} split from {data_dir}...")
    compute_word_latent_mapping_on_dataset(
        model=model,
        tokenizer=tokenizer,
        split_name=args.split_name,
        checkpoint_path=args.checkpoint_path,
        data_dir=data_dir,
        num_thoughts=num_thoughts,
        seed=args.seed
    )
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main() 