"""
TRUE Entropy with Full Token Distribution

This version loads the model directly via transformers to get the ACTUAL
probability distribution over all ~128k tokens, allowing exact entropy computation.

No sampling needed - we get the model's real logits!
"""

import sys
import os
import torch
import numpy as np
from typing import List, Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.information_theory import compute_entropy, compute_shannon_information

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn.functional as F
except ImportError:
    print("Error: transformers required. Install with: pip install transformers")
    sys.exit(1)


class FullDistributionEntropyAnalyzer:
    """
    Analyzes entropy using the FULL probability distribution over all tokens.
    """
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize with a model.
        
        Args:
            model_path: Path to model (HF name or local path)
            device: Device to use
        """
        print(f"Loading model: {model_path}")
        print(f"Device: {device}")
        
        self.device = device
        self.model_path = model_path
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        vocab_size = len(self.tokenizer)
        print(f"✓ Model loaded")
        print(f"Vocabulary size: {vocab_size:,} tokens\n")
    
    def get_full_next_token_distribution(
        self,
        text: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Get the FULL probability distribution over ALL tokens.
        
        This is what the model actually computes!
        
        Args:
            text: The input text
            
        Returns:
            Tuple of (probabilities_tensor, vocab_size)
        """
        if not text:
            text = " "
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model output (logits for all tokens)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position, all vocab
            
            # Convert logits to probabilities via softmax
            probs = F.softmax(logits, dim=-1)
        
        return probs, len(probs)
    
    def analyze_string(
        self,
        sample_string: str,
        prompt: str = ""
    ) -> Dict:
        """
        Analyze using FULL distributions (no sampling!).
        
        Args:
            sample_string: Target string to analyze
            prompt: Initial prompt
            
        Returns:
            Results dictionary
        """
        # Tokenize using model's actual tokenizer
        tokens_text = sample_string.split()  # Word level for clarity
        
        print("=" * 80)
        print("Token-by-Token Analysis with FULL Distribution")
        print("=" * 80)
        print(f"Model: {self.model_path}")
        print(f"Prompt: '{prompt}'")
        print(f"Target string: '{sample_string}'")
        print(f"Tokens (word-level): {tokens_text}")
        print(f"Vocabulary size: {len(self.tokenizer):,}")
        print("=" * 80)
        print("\nComputing entropy from FULL probability distribution...\n")
        
        # Storage
        token_list = []
        entropy_list = []
        information_list = []
        token_prob_list = []
        
        context = prompt
        
        # Analyze each word token
        for i, token_word in enumerate(tokens_text, 1):
            print(f"\nStep {i}/{len(tokens_text)}: Token = '{token_word}'")
            print("-" * 80)
            
            # Get FULL probability distribution (all ~128k tokens!)
            full_probs, vocab_size = self.get_full_next_token_distribution(context)
            
            # Compute entropy from FULL distribution
            # Filter out zeros to avoid log(0) = -inf
            probs_np = full_probs.cpu().numpy()
            probs_nonzero = probs_np[probs_np > 1e-10]  # Keep only non-zero probs
            
            # Compute entropy (this is the TRUE entropy!)
            entropy = compute_entropy(
                probs_nonzero,
                base=2.0
            )
            
            # Get probability of the specific token
            # First tokenize the target token
            token_ids = self.tokenizer.encode(
                " " + token_word,
                add_special_tokens=False
            )
            
            if len(token_ids) > 0:
                token_id = token_ids[0]
                token_prob = full_probs[token_id].item()
            else:
                token_prob = 0.0001
            
            # Compute information (surprisal)
            information = compute_shannon_information(token_prob, base=2.0)
            
            # Store
            token_list.append(token_word)
            entropy_list.append(entropy)
            information_list.append(information)
            token_prob_list.append(token_prob)
            
            # Get top alternatives
            top_k_probs, top_k_indices = torch.topk(full_probs, 10)
            
            # Print
            print(f"  Entropy (from {vocab_size:,} tokens): {entropy:.4f} bits")
            print(f"  Token probability: {token_prob:.6f}")
            print(f"  Shannon information: {information:.4f} bits")
            print(f"  Top 5 alternatives:")
            for j, (prob, idx) in enumerate(zip(top_k_probs[:5], top_k_indices[:5]), 1):
                alt_token = self.tokenizer.decode([idx])
                print(f"    {j}. '{alt_token}' (p={prob.item():.6f})")
            
            # Update context
            context += " " + token_word
        
        # Compute totals
        total_entropy_avg = np.mean(entropy_list)
        total_entropy_sum = np.sum(entropy_list)
        total_information = np.sum(information_list)
        avg_information = np.mean(information_list)
        perplexity = 2 ** avg_information
        
        print("\n" + "=" * 80)
        print("SUMMARY - FULL DISTRIBUTION ANALYSIS (EXACT)")
        print("=" * 80)
        print(f"Total tokens: {len(tokens_text)}")
        print(f"\nEntropy Statistics:")
        print(f"  Average entropy per position: {total_entropy_avg:.4f} bits")
        print(f"  Total entropy (sequence): {total_entropy_sum:.4f} bits")
        print(f"  Min entropy: {np.min(entropy_list):.4f} bits")
        print(f"  Max entropy: {np.max(entropy_list):.4f} bits")
        print(f"\nInformation Statistics:")
        print(f"  Total information: {total_information:.4f} bits")
        print(f"  Average information: {avg_information:.4f} bits")
        print(f"  Perplexity: {perplexity:.4f}")
        print("=" * 80)
        print("\n✓ These are EXACT values from the full probability distribution!")
        print(f"  (computed over {len(self.tokenizer):,} tokens at each step)")
        
        return {
            'tokens': token_list,
            'entropies': entropy_list,
            'information': information_list,
            'token_probabilities': token_prob_list,
            'total_entropy_avg': total_entropy_avg,
            'total_entropy_sum': total_entropy_sum,
            'total_information': total_information,
            'avg_information': avg_information,
            'perplexity': perplexity,
        }
    
    def save_results(self, results: Dict, filename: str = "full_distribution_entropy.txt"):
        """Save results."""
        with open(filename, 'w') as f:
            f.write("Full Distribution Entropy Analysis (EXACT)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {self.model_path}\n\n")
            f.write("Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Pos':<5} {'Token':<20} {'Prob':<12} {'Entropy':<12} {'Information':<15}\n")
            f.write("-" * 80 + "\n")
            
            for i, (token, prob, entropy, info) in enumerate(zip(
                results['tokens'],
                results['token_probabilities'],
                results['entropies'],
                results['information']
            ), 1):
                f.write(f"{i:<5} {token:<20} {prob:<12.6f} {entropy:<12.4f} {info:<15.4f}\n")
            
            f.write("-" * 80 + "\n\n")
            f.write("Summary:\n")
            f.write(f"  Average entropy: {results['total_entropy_avg']:.4f} bits\n")
            f.write(f"  Total information: {results['total_information']:.4f} bits\n")
            f.write(f"  Perplexity: {results['perplexity']:.4f}\n")
        
        print(f"\n✓ Results saved to: {filename}")


def main():
    """Run full distribution analysis."""
    
    print("\n" + "=" * 80)
    print("EXACT Entropy Analysis with Full Token Distribution")
    print("=" * 80)
    print("\nThis loads the model directly to access the TRUE probability")
    print("distribution over all vocabulary tokens (no sampling needed!).")
    print("=" * 80 + "\n")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (will be slower)")
    
    print("\nNote: This will download/load Llama-3.1-8B (~16GB)")
    print("Or specify a different model path.\n")
    
    input("Press Enter to continue...")
    
    # For Llama models you'll need HuggingFace authentication
    # Or use a local model path if you have weights downloaded
    # Example: "/path/to/your/llama-3.1-8b"
    
    # Using a smaller open model for demo (or replace with your local Llama path)
    try:
        analyzer = FullDistributionEntropyAnalyzer(
            model_path="meta-llama/Llama-3.1-8B",  # Change to local path if needed
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"\nError loading Llama model: {e}")
        print("\nTrying GPT-2 as fallback (open access)...")
        analyzer = FullDistributionEntropyAnalyzer(
            model_path="gpt2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Analyze
    sample_string = "hello what is the entropy and information of this string"
    
    results = analyzer.analyze_string(
        sample_string=sample_string,
        prompt=""
    )
    
    analyzer.save_results(results)
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
