"""
REAL Entropy and Information Computation with Transformers

This version uses the transformers library to get actual token probabilities
from the model, allowing real entropy and information calculations.

This approach:
1. Loads the model directly via transformers (not through Ollama)
2. Gets real logits and probabilities at each generation step
3. Computes actual entropy and information values
4. Can use any HuggingFace model or local model

Note: This requires more VRAM but provides accurate probability distributions.
"""

import sys
import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.information_theory import compute_entropy, compute_shannon_information

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn.functional as F
except ImportError:
    print("Error: transformers library required for real probability extraction")
    print("Install with: pip install transformers torch")
    sys.exit(1)


class RealEntropyAnalyzer:
    """
    Analyzes entropy and information using real model probabilities.
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to run on ('cuda' or 'cpu')
        """
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ Model loaded successfully\n")
    
    def get_next_token_probabilities(
        self, 
        prompt: str, 
        context: str,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get probability distribution over next tokens.
        
        Args:
            prompt: Initial prompt
            context: Generated text so far
            top_k: Number of top tokens to return
            
        Returns:
            Tuple of (full_probs_tensor, top_k_token_dict)
        """
        # Combine prompt and context
        full_text = prompt + context if context else prompt
        
        # Handle empty string - use BOS token
        if not full_text:
            full_text = self.tokenizer.bos_token if self.tokenizer.bos_token else " "
        
        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
        
        # Get top-k tokens
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        # Create dictionary of top tokens
        top_tokens = {}
        for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
            token = self.tokenizer.decode([idx])
            top_tokens[token] = float(prob)
        
        return probs, top_tokens
    
    def get_token_probability(
        self,
        token: str,
        prompt: str,
        context: str
    ) -> float:
        """
        Get the probability of a specific token.
        
        Args:
            token: The token to get probability for
            prompt: Initial prompt
            context: Generated text so far
            
        Returns:
            Probability of the token
        """
        # Get full probability distribution
        probs, _ = self.get_next_token_probabilities(prompt, context, top_k=1)
        
        # Tokenize the target token
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        
        if len(token_ids) == 0:
            return 0.0
        
        # Get probability of first token ID (simplified)
        token_id = token_ids[0]
        prob = probs[token_id].item()
        
        return prob
    
    def analyze_string_with_real_probabilities(
        self,
        sample_string: str,
        prompt: str = "",
        use_word_tokens: bool = False
    ) -> Dict:
        """
        Analyze entropy and information using REAL model probabilities.
        
        Args:
            sample_string: The target string to analyze
            prompt: Initial prompt
            use_word_tokens: If False, use model's actual tokenization
            
        Returns:
            Dictionary containing analysis results
        """
        # Tokenize the sample string
        if use_word_tokens:
            tokens = sample_string.split()
        else:
            # Use model's actual tokenization
            token_ids = self.tokenizer.encode(sample_string, add_special_tokens=False)
            tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        print("=" * 80)
        print("Token-by-Token Analysis with REAL Probabilities")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Prompt: '{prompt}'")
        print(f"Target string: '{sample_string}'")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print("=" * 80)
        
        # Storage for results
        token_list = []
        entropy_list = []
        information_list = []
        token_prob_list = []
        top_alternatives = []
        
        # Current context
        context = ""
        
        # Analyze each token
        for i, token in enumerate(tokens, 1):
            print(f"\nStep {i}/{len(tokens)}: Token = '{token}'")
            print("-" * 80)
            
            # Get probability distribution
            full_probs, top_tokens = self.get_next_token_probabilities(
                prompt, context, top_k=10
            )
            
            # Compute entropy of distribution (filter out zeros to avoid nan)
            probs_numpy = full_probs.cpu().numpy()
            probs_nonzero = probs_numpy[probs_numpy > 1e-10]  # Filter near-zero probs
            entropy = compute_entropy(
                probs_nonzero,
                base=2.0
            )
            
            # Get probability of the actual token
            token_prob = self.get_token_probability(token, prompt, context)
            
            # Compute Shannon information
            if token_prob > 0:
                information = compute_shannon_information(token_prob, base=2.0)
            else:
                information = float('inf')  # Token has zero probability
            
            # Store results
            token_list.append(token)
            entropy_list.append(entropy)
            information_list.append(information)
            token_prob_list.append(token_prob)
            top_alternatives.append(top_tokens)
            
            # Print results
            print(f"  Entropy of distribution: {entropy:.4f} bits")
            print(f"  Token probability: {token_prob:.6f}")
            print(f"  Shannon information: {information:.4f} bits")
            print(f"  Top 5 alternative tokens:")
            for j, (alt_token, alt_prob) in enumerate(list(top_tokens.items())[:5], 1):
                print(f"    {j}. '{alt_token}' (p={alt_prob:.6f})")
            
            # Update context
            context += token
        
        # Compute totals
        total_entropy_avg = np.mean(entropy_list)
        total_entropy_sum = np.sum(entropy_list)
        total_information = np.sum(information_list)
        avg_information = np.mean(information_list)
        
        # Perplexity (exp of average information)
        perplexity = 2 ** avg_information
        
        print("\n" + "=" * 80)
        print("SUMMARY - REAL PROBABILITY ANALYSIS")
        print("=" * 80)
        print(f"Total tokens: {len(tokens)}")
        print(f"\nEntropy Statistics:")
        print(f"  Average entropy per position: {total_entropy_avg:.4f} bits")
        print(f"  Total entropy (sum): {total_entropy_sum:.4f} bits")
        print(f"  Min entropy: {np.min(entropy_list):.4f} bits")
        print(f"  Max entropy: {np.max(entropy_list):.4f} bits")
        print(f"\nInformation Statistics:")
        print(f"  Total information: {total_information:.4f} bits")
        print(f"  Average information per token: {avg_information:.4f} bits")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Min information: {np.min(information_list):.4f} bits")
        print(f"  Max information: {np.max(information_list):.4f} bits")
        print("=" * 80)
        
        return {
            'tokens': token_list,
            'entropies': entropy_list,
            'information': information_list,
            'token_probabilities': token_prob_list,
            'top_alternatives': top_alternatives,
            'total_entropy_avg': total_entropy_avg,
            'total_entropy_sum': total_entropy_sum,
            'total_information': total_information,
            'avg_information': avg_information,
            'perplexity': perplexity,
            'prompt': prompt,
            'sample_string': sample_string,
        }
    
    def save_results(self, results: Dict, filename: str = "real_entropy_results.txt"):
        """Save detailed results to file."""
        with open(filename, 'w') as f:
            f.write("Token-by-Token Entropy and Information Analysis (REAL PROBABILITIES)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Sample String: {results['sample_string']}\n")
            f.write(f"Prompt: {results['prompt']}\n\n")
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
            f.write(f"  Total tokens: {len(results['tokens'])}\n")
            f.write(f"  Average entropy: {results['total_entropy_avg']:.4f} bits\n")
            f.write(f"  Total information: {results['total_information']:.4f} bits\n")
            f.write(f"  Perplexity: {results['perplexity']:.4f}\n")
        
        print(f"\n✓ Results saved to: {filename}")


def main():
    """
    Run real entropy analysis.
    
    NOTE: This uses a small model by default (Llama-3.2-1B).
    For Llama-3.3, you would need to:
    1. Have the model downloaded locally
    2. Or use the HuggingFace version (requires authentication)
    3. Adjust model_name parameter
    """
    
    print("\n" + "=" * 80)
    print("REAL Entropy and Information Analysis")
    print("=" * 80)
    print("\nThis example uses the transformers library to get real probabilities.")
    print("It will download a small model (~2.5GB) on first run.")
    print("=" * 80 + "\n")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available - using CPU (will be slower)")
    
    input("\nPress Enter to continue...")
    
    # Initialize with a small model (you can change this)
    # Using GPT-2 (no authentication needed, ~500MB)
    analyzer = RealEntropyAnalyzer(
        model_name="gpt2",  # Open-access model for demo
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Sample string
    sample_string = "hello what is the entropy and information of this string"
    
    # Run analysis
    results = analyzer.analyze_string_with_real_probabilities(
        sample_string=sample_string,
        prompt="",
        use_word_tokens=False  # Use model's actual tokenization
    )
    
    # Save results
    analyzer.save_results(results, "real_entropy_analysis.txt")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
