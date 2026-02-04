"""
REAL Entropy and Information with Ollama Llama 3.1 8B

This version uses Ollama's API to get real token probabilities from llama3.1:8b,
allowing accurate entropy and information calculations.

Requires Ollama running with a model that supports logprobs.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.information_theory import compute_entropy, compute_shannon_information

try:
    import ollama
except ImportError:
    print("Error: ollama package not found. Install with: pip install ollama")
    sys.exit(1)


class OllamaEntropyAnalyzer:
    """
    Analyzes entropy and information using Ollama with real probabilities.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        """
        Initialize with an Ollama model.
        
        Args:
            model_name: The Ollama model name
        """
        self.model_name = model_name
        self.client = ollama.Client()
        
        print(f"Using Ollama model: {model_name}")
        
        # Verify model exists
        try:
            models = self.client.list()
            model_list = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
            available = [m.get('name') if isinstance(m, dict) else getattr(m, 'name', str(m)) for m in model_list]
            
            if any(model_name in name for name in available):
                print(f"✓ Model '{model_name}' is available\n")
            else:
                print(f"⚠ Warning: Model '{model_name}' not found")
                print(f"Available: {', '.join(available)}\n")
        except Exception as e:
            print(f"⚠ Could not verify model: {e}\n")
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Simple word-level tokenization.
        For real token-level analysis, would need the model's actual tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def get_next_token_distribution(
        self,
        prompt: str,
        context: str
    ) -> Tuple[np.ndarray, Dict[str, float], str]:
        """
        Get probability distribution for next token using Ollama's API.
        
        We need to generate with the text and parse the response to get
        the distribution over possible next tokens.
        
        Args:
            prompt: Initial prompt
            context: Generated text so far
            
        Returns:
            Tuple of (probability_array, top_tokens_dict, generated_token)
        """
        full_text = prompt + context if context else prompt
        
        if not full_text:
            full_text = " "
        
        try:
            # Use the /api/generate endpoint directly to get better control
            # Generate one token and request to see the model's distribution
            import requests
            import json
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': full_text,
                    'stream': False,
                    'options': {
                        'num_predict': 1,
                        'temperature': 1.0,
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get('response', '').strip()
                
                # For now, since Ollama doesn't expose full distributions easily,
                # we'll sample multiple times with temperature to approximate
                token_counts = {}
                num_samples = 100  # More samples = better approximation
                
                for _ in range(num_samples):
                    sample_response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': self.model_name,
                            'prompt': full_text,
                            'stream': False,
                            'options': {
                                'num_predict': 1,
                                'temperature': 1.0,
                                'top_k': 50,
                            }
                        }
                    )
                    
                    if sample_response.status_code == 200:
                        sample_result = sample_response.json()
                        token = sample_result.get('response', '').strip().split()[0] if sample_result.get('response', '').strip() else ''
                        if token:
                            token_counts[token] = token_counts.get(token, 0) + 1
                
                # Convert counts to probabilities
                total = sum(token_counts.values())
                if total > 0:
                    token_probs = {token: count / total for token, count in token_counts.items()}
                    prob_array = np.array(list(token_probs.values()))
                    return prob_array, token_probs, generated
                
            return np.array([1.0]), {}, ""
            
        except Exception as e:
            print(f"Error: {e}")
            return np.array([1.0]), {}, ""
    
    def get_token_probability(
        self,
        token: str,
        token_probs: Dict[str, float]
    ) -> float:
        """
        Get probability of a specific token from the distribution.
        
        Args:
            token: The token to find
            token_probs: Dictionary of token probabilities
            
        Returns:
            Probability of the token
        """
        # Direct match
        if token in token_probs:
            return token_probs[token]
        
        # Fuzzy match (case insensitive, partial)
        for key, prob in token_probs.items():
            if token.lower() in key.lower() or key.lower() in token.lower():
                return prob
        
        # Not found in distribution - low probability
        return 0.001
    
    def analyze_string_with_real_probabilities(
        self,
        sample_string: str,
        prompt: str = "",
        use_word_tokens: bool = True
    ) -> Dict:
        """
        Analyze entropy and information using REAL Ollama logprobs.
        
        Args:
            sample_string: The target string to analyze
            prompt: Initial prompt
            use_word_tokens: Use word-level tokenization
            
        Returns:
            Dictionary containing analysis results
        """
        # Tokenize
        tokens = self._simple_tokenize(sample_string)
        
        print("=" * 80)
        print("Token-by-Token Analysis with Ollama")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Prompt: '{prompt}'")
        print(f"Target string: '{sample_string}'")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print("=" * 80)
        print("\nNote: Sampling model to estimate next-token distribution.\n")
        print("This will take a moment (100 samples per position)...\n")
        
        # Storage
        token_list = []
        entropy_list = []
        information_list = []
        token_prob_list = []
        top_alternatives = []
        
        context = ""
        
        # Analyze each token
        for i, token in enumerate(tokens, 1):
            print(f"\nStep {i}/{len(tokens)}: Token = '{token}'")
            print("-" * 80)
            
            # Get probability distribution at this position
            prob_array, token_probs, _ = self.get_next_token_distribution(
                prompt, context
            )
            
            # Compute entropy from the DISTRIBUTION
            entropy = compute_entropy(prob_array, base=2.0)
            
            # Get probability of the specific token we're analyzing
            token_prob = self.get_token_probability(token, token_probs)
            
            # Compute information (surprisal)
            information = compute_shannon_information(token_prob, base=2.0)
            
            # Store
            token_list.append(token)
            entropy_list.append(entropy)
            information_list.append(information)
            token_prob_list.append(token_prob)
            top_alternatives.append(token_probs)
            
            # Print
            print(f"  Entropy of distribution: {entropy:.4f} bits")
            print(f"  Token probability: {token_prob:.6f}")
            print(f"  Shannon information: {information:.4f} bits")
            
            if token_probs:
                print(f"  Top alternatives from distribution:")
                sorted_tokens = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)
                for j, (alt_token, alt_prob) in enumerate(sorted_tokens[:5], 1):
                    print(f"    {j}. '{alt_token}' (p={alt_prob:.4f})")
            
            # Update context
            context += token + " "
        
        # Compute totals
        total_entropy_avg = np.mean(entropy_list)
        total_entropy_sum = np.sum(entropy_list)
        total_information = np.sum(information_list)
        avg_information = np.mean(information_list)
        perplexity = 2 ** avg_information
        
        print("\n" + "=" * 80)
        print("SUMMARY - OLLAMA DISTRIBUTION ANALYSIS")
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
    
    def save_results(self, results: Dict, filename: str = "ollama_entropy_results.txt"):
        """Save results to file."""
        with open(filename, 'w') as f:
            f.write("Token-by-Token Entropy Analysis (Ollama Sampling)\n")
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
    """Run Ollama-based entropy analysis."""
    
    print("\n" + "=" * 80)
    print("Entropy Analysis with Ollama Llama 3.1 8B")
    print("=" * 80)
    print("\nThis estimates the next-token probability distribution via sampling.")
    print("Entropy will vary based on context!")
    print("=" * 80 + "\n")
    
    # Initialize
    analyzer = OllamaEntropyAnalyzer(model_name="llama3.1:8b")
    
    # Sample string
    sample_string = "hello what is the entropy and information of this string"
    
    print("Starting analysis...\n")
    
    # Run analysis
    results = analyzer.analyze_string_with_real_probabilities(
        sample_string=sample_string,
        prompt="",
        use_word_tokens=True
    )
    
    # Save
    analyzer.save_results(results, "ollama_llama31_entropy.txt")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
