"""
Entropy and Information Computation with Token-by-Token Intervention

This example demonstrates how to:
1. Force specific tokens during generation (intervention)
2. Compute entropy of the probability distribution at each step
3. Compute Shannon information (surprisal) of each chosen token
4. Calculate total entropy and information for the entire sequence

Note: Ollama's standard API doesn't expose token probabilities by default.
This example shows the framework and uses mock probabilities for demonstration.
For real probability extraction, you would need to use the raw Ollama API or
integrate with a library like transformers that exposes logits.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.information_theory import compute_entropy, compute_shannon_information
import ollama


class EntropyInterventionAnalyzer:
    """
    Analyzes entropy and information during token-by-token generation with intervention.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        """
        Initialize the analyzer.
        
        Args:
            model_name: The Ollama model to use
        """
        self.model_name = model_name
        self.client = ollama.Client()
        
    def _get_token_probabilities(self, prompt: str, context: str) -> Dict[str, float]:
        """
        Get probability distribution over next tokens.
        
        NOTE: This is a MOCK function for demonstration purposes.
        Ollama's standard API doesn't expose token probabilities.
        
        To implement this for real:
        1. Use Ollama's raw API with logprobs enabled (if available)
        2. Use transformers library directly with the model
        3. Use a custom model server that exposes logits
        
        Args:
            prompt: The initial prompt
            context: The text generated so far
            
        Returns:
            Dictionary mapping tokens to probabilities (mock data)
        """
        # MOCK: Generate synthetic probability distribution
        # In reality, you would get this from the model's logits
        
        # For demonstration, create a realistic-looking distribution
        full_context = prompt + context if context else prompt
        
        # Mock: Generate random probabilities for demonstration
        # The actual implementation would query the model
        np.random.seed(len(full_context))  # Deterministic for demo
        
        # Simulate a realistic probability distribution
        # Top tokens have higher probabilities
        num_tokens = 50000  # Vocabulary size (approximate)
        
        # Create a power-law distribution (realistic for language models)
        ranks = np.arange(1, 101)  # Top 100 tokens
        raw_probs = 1.0 / (ranks ** 1.5)
        probs = raw_probs / raw_probs.sum()
        
        # Create mock token distribution
        token_probs = {
            f"token_{i}": float(probs[i-1]) 
            for i in range(1, 101)
        }
        
        return token_probs
    
    def _get_token_probability(self, token: str, prompt: str, context: str) -> float:
        """
        Get the probability of a specific token being generated.
        
        NOTE: This is a MOCK function for demonstration.
        
        Args:
            token: The token to get probability for
            prompt: The initial prompt
            context: The text generated so far
            
        Returns:
            Probability of the token (mock data)
        """
        # MOCK: Return a realistic probability
        # In reality, this would come from the model's output distribution
        
        # For common tokens, assign higher probability
        common_tokens = ['hello', 'what', 'is', 'the', 'of', 'this', 'and', 'to']
        
        if token.lower() in common_tokens:
            return np.random.uniform(0.05, 0.15)  # 5-15% probability
        else:
            return np.random.uniform(0.01, 0.05)  # 1-5% probability
    
    def analyze_string_with_intervention(
        self,
        sample_string: str,
        prompt: str = "",
        use_real_tokenization: bool = False
    ) -> Dict:
        """
        Analyze entropy and information for each token in a string.
        
        This simulates the generation process where:
        1. At each step, we get the probability distribution over next tokens
        2. We intervene and force a specific token from sample_string
        3. We compute entropy of the distribution and information of the chosen token
        
        Args:
            sample_string: The target string to analyze
            prompt: Initial prompt (can be empty)
            use_real_tokenization: If True, use model's tokenizer (not implemented in Ollama)
            
        Returns:
            Dictionary containing tokens, entropies, information values, and totals
        """
        # For simplicity, split by spaces (mock tokenization)
        # Real implementation would use the model's actual tokenizer
        if use_real_tokenization:
            print("⚠ Warning: Real tokenization not available with Ollama API")
            print("Using simple space-based tokenization for demonstration\n")
        
        tokens = sample_string.split()
        
        # Storage for results
        token_list = []
        entropy_list = []
        information_list = []
        
        # Current context (what has been generated so far)
        context = ""
        
        print("=" * 80)
        print("Token-by-Token Analysis with Intervention")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Target string: '{sample_string}'")
        print(f"Number of tokens: {len(tokens)}")
        print("=" * 80)
        
        # Analyze each token
        for i, token in enumerate(tokens, 1):
            print(f"\nStep {i}/{len(tokens)}: Token = '{token}'")
            print("-" * 80)
            
            # Get probability distribution over next tokens
            token_probs = self._get_token_probabilities(prompt, context)
            
            # Convert to probability array for entropy calculation
            prob_values = np.array(list(token_probs.values()))
            
            # Compute entropy of the distribution
            entropy = compute_entropy(prob_values, base=2.0)
            
            # Get probability of the token we're forcing
            token_prob = self._get_token_probability(token, prompt, context)
            
            # Compute Shannon information (surprisal) of this token
            # I(x) = -log2(P(x))
            information = compute_shannon_information(token_prob, base=2.0)
            
            # Store results
            token_list.append(token)
            entropy_list.append(entropy)
            information_list.append(information)
            
            # Print step results
            print(f"  Entropy of distribution: {entropy:.4f} bits")
            print(f"  Token probability: {token_prob:.6f}")
            print(f"  Shannon information: {information:.4f} bits")
            print(f"  Context so far: '{context}{token}'")
            
            # Update context with the forced token
            context += token + " "
        
        # Compute totals
        # Total entropy: Average entropy across all positions
        total_entropy_avg = np.mean(entropy_list)
        total_entropy_sum = np.sum(entropy_list)
        
        # Total information: Sum of all surprisal values
        # This represents the total "surprise" or information content of the sequence
        total_information = np.sum(information_list)
        
        # Average information per token
        avg_information = np.mean(information_list)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
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
        print(f"  Min information: {np.min(information_list):.4f} bits")
        print(f"  Max information: {np.max(information_list):.4f} bits")
        print("=" * 80)
        
        # Return structured results
        return {
            'tokens': token_list,
            'entropies': entropy_list,
            'information': information_list,
            'total_entropy_avg': total_entropy_avg,
            'total_entropy_sum': total_entropy_sum,
            'total_information': total_information,
            'avg_information': avg_information,
            'prompt': prompt,
            'sample_string': sample_string,
        }
    
    def plot_results(self, results: Dict):
        """
        Plot entropy and information across token positions.
        
        Args:
            results: Results dictionary from analyze_string_with_intervention
        """
        try:
            import matplotlib.pyplot as plt
            
            tokens = results['tokens']
            entropies = results['entropies']
            information = results['information']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot entropy
            ax1.plot(range(1, len(tokens) + 1), entropies, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Token Position')
            ax1.set_ylabel('Entropy (bits)')
            ax1.set_title('Entropy of Token Distribution at Each Position')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=np.mean(entropies), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(entropies):.2f} bits')
            ax1.legend()
            
            # Plot information
            ax2.plot(range(1, len(tokens) + 1), information, 'g-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Token Position')
            ax2.set_ylabel('Information (bits)')
            ax2.set_title('Shannon Information (Surprisal) of Each Token')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=np.mean(information), color='r', linestyle='--',
                       label=f'Mean: {np.mean(information):.2f} bits')
            ax2.legend()
            
            # Set token labels on x-axis (if not too many)
            if len(tokens) <= 20:
                for ax in [ax1, ax2]:
                    ax.set_xticks(range(1, len(tokens) + 1))
                    ax.set_xticklabels(tokens, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('entropy_information_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: entropy_information_analysis.png")
            plt.close()
            
        except ImportError:
            print("\n⚠ matplotlib not available for plotting")
        except Exception as e:
            print(f"\n⚠ Error creating plot: {e}")


def main():
    """Run the entropy and information analysis example."""
    
    print("\n" + "=" * 80)
    print("Entropy and Information Analysis with Token Intervention")
    print("=" * 80)
    print("\nNOTE: This example uses MOCK probabilities for demonstration.")
    print("To use real probabilities, you need to:")
    print("  1. Access model logits directly (requires transformers library)")
    print("  2. Use Ollama raw API with logprobs (if available)")
    print("  3. Set up a custom model server that exposes probabilities")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = EntropyInterventionAnalyzer(model_name="llama3.1:8b")
    
    # Sample string to analyze
    sample_string = "hello what is the entropy and information of this string"
    
    # Run analysis with empty prompt
    results = analyzer.analyze_string_with_intervention(
        sample_string=sample_string,
        prompt=""
    )
    
    # Save results to file
    print("\n" + "=" * 80)
    print("Saving detailed results...")
    print("=" * 80)
    
    output_file = "entropy_information_results.txt"
    with open(output_file, 'w') as f:
        f.write("Token-by-Token Entropy and Information Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sample String: {sample_string}\n")
        f.write(f"Prompt: {results['prompt']}\n\n")
        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Position':<10} {'Token':<20} {'Entropy (bits)':<15} {'Information (bits)':<20}\n")
        f.write("-" * 80 + "\n")
        
        for i, (token, entropy, info) in enumerate(zip(
            results['tokens'], 
            results['entropies'], 
            results['information']
        ), 1):
            f.write(f"{i:<10} {token:<20} {entropy:<15.4f} {info:<20.4f}\n")
        
        f.write("-" * 80 + "\n\n")
        f.write("Summary:\n")
        f.write(f"  Total tokens: {len(results['tokens'])}\n")
        f.write(f"  Average entropy: {results['total_entropy_avg']:.4f} bits\n")
        f.write(f"  Total entropy (sum): {results['total_entropy_sum']:.4f} bits\n")
        f.write(f"  Total information: {results['total_information']:.4f} bits\n")
        f.write(f"  Average information: {results['avg_information']:.4f} bits\n")
    
    print(f"✓ Detailed results saved to: {output_file}")
    
    # Create visualization
    analyzer.plot_results(results)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nNext steps to use real probabilities:")
    print("1. Use transformers library to load the model directly")
    print("2. Extract logits at each generation step")
    print("3. Convert logits to probabilities using softmax")
    print("4. Replace _get_token_probabilities() with real implementation")


if __name__ == "__main__":
    main()
