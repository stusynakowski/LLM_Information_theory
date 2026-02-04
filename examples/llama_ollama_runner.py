"""
Llama 3.3 Runner via Ollama

This script provides a template for running Llama 3.3 through Ollama with
support for information-theoretic analysis and intervention capabilities.

The script is designed to be easily extensible for:
- Computing entropy of token probability distributions
- Measuring information content
- Intervening in the generation process
- Tracking generation statistics
"""

import sys
import os
from typing import Dict, List, Optional, Callable, Tuple
import json
import numpy as np

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ollama
except ImportError:
    print("Error: ollama package not found. Install with: pip install ollama")
    print("Also ensure ollama is running: ollama serve")
    sys.exit(1)

from src.information_theory import compute_entropy, compute_shannon_information


class OllamaLlamaRunner:
    """
    A wrapper for running Llama 3.3 via Ollama with information theory hooks.
    
    This class provides:
    - Basic text generation
    - Streaming generation with per-token intervention
    - Entropy and information computation (when available)
    - Generation statistics tracking
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        """
        Initialize the Ollama Llama runner.
        
        Args:
            model_name: The name of the model in Ollama (default: llama3.1:8b)
        """
        self.model_name = model_name
        self.client = ollama.Client()
        self.generation_history = []
        
        # Verify model is available
        self._verify_model()
    
    def _verify_model(self):
        """Check if the model is available in Ollama."""
        try:
            models = self.client.list()
            # Handle both dict and object response formats
            model_list = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
            available_models = [m.get('name') if isinstance(m, dict) else getattr(m, 'name', str(m)) for m in model_list]
            
            # Check if our model is in the list
            model_found = any(self.model_name in name for name in available_models)
            
            if model_found:
                print(f"✓ Model '{self.model_name}' is available")
            else:
                print(f"⚠ Warning: Model '{self.model_name}' not found in Ollama")
                print(f"Available models: {', '.join(available_models)}")
                print(f"\nTo pull the model, run: ollama pull {self.model_name}")
        except Exception as e:
            print(f"⚠ Warning: Could not verify model availability: {e}")
            print("Make sure Ollama is running (ollama serve)")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            stop_sequences: List of sequences that stop generation
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with generation results and metadata
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop_sequences:
            options["stop"] = stop_sequences
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                stream=False
            )
            
            result = {
                "prompt": prompt,
                "response": response['message']['content'],
                "model": response.get('model', self.model_name),
                "total_duration": response.get('total_duration', 0) / 1e9,  # Convert to seconds
                "prompt_eval_count": response.get('prompt_eval_count', 0),
                "eval_count": response.get('eval_count', 0),
            }
            
            self.generation_history.append(result)
            return result
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        intervention_callback: Optional[Callable[[str, str], Optional[str]]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Generate text with streaming output and optional intervention.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            intervention_callback: Optional function called for each token
                                  Signature: fn(full_text_so_far, new_chunk) -> Optional[stop_signal]
                                  Return "STOP" to halt generation
            system_prompt: Optional system prompt
            verbose: Print tokens as they are generated
            
        Returns:
            Dictionary with generation results
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        
        full_response = ""
        tokens_generated = 0
        should_stop = False
        
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if should_stop:
                    break
                
                if 'message' in chunk and 'content' in chunk['message']:
                    token_text = chunk['message']['content']
                    full_response += token_text
                    tokens_generated += 1
                    
                    if verbose:
                        print(token_text, end='', flush=True)
                    
                    # Call intervention callback if provided
                    if intervention_callback:
                        result = intervention_callback(full_response, token_text)
                        if result == "STOP":
                            should_stop = True
                            if verbose:
                                print("\n[Generation stopped by intervention]")
            
            if verbose:
                print()  # New line after generation
            
            result = {
                "prompt": prompt,
                "response": full_response,
                "model": self.model_name,
                "tokens_generated": tokens_generated,
                "stopped_by_intervention": should_stop
            }
            
            self.generation_history.append(result)
            return result
            
        except Exception as e:
            print(f"\nError during streaming generation: {e}")
            raise
    
    def compute_response_statistics(self, text: str) -> Dict:
        """
        Compute basic statistics about generated text.
        
        This is a placeholder for more sophisticated information-theoretic
        analysis. Extend this method to compute:
        - Token-level entropy
        - Surprisal values
        - Information content
        
        Args:
            text: The generated text
            
        Returns:
            Dictionary of statistics
        """
        words = text.split()
        chars = len(text)
        
        # Basic statistics
        stats = {
            "word_count": len(words),
            "char_count": chars,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "unique_words": len(set(words)),
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
        }
        
        # TODO: Add entropy computation when token probabilities are available
        # Note: Ollama's standard API doesn't expose token probabilities by default
        # You would need to use the raw API or modify the model server for this
        
        return stats
    
    def intervene_on_keyword(
        self,
        prompt: str,
        keywords: List[str],
        action: str = "stop",
        **generation_kwargs
    ) -> Dict:
        """
        Generate text and intervene when specific keywords appear.
        
        Args:
            prompt: The input prompt
            keywords: List of keywords to watch for
            action: Action to take ("stop" or "alert")
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Generation result with intervention information
        """
        keywords_lower = [k.lower() for k in keywords]
        interventions = []
        
        def intervention_fn(full_text: str, new_chunk: str):
            full_text_lower = full_text.lower()
            
            for keyword in keywords_lower:
                if keyword in full_text_lower and keyword not in [i['keyword'] for i in interventions]:
                    interventions.append({
                        "keyword": keyword,
                        "position": len(full_text),
                        "context": full_text[-100:]  # Last 100 chars
                    })
                    
                    if action == "stop":
                        return "STOP"
                    elif action == "alert":
                        print(f"\n⚠ Keyword detected: '{keyword}'")
            
            return None
        
        result = self.generate_streaming(
            prompt,
            intervention_callback=intervention_fn,
            **generation_kwargs
        )
        
        result['interventions'] = interventions
        return result
    
    def get_history(self) -> List[Dict]:
        """Return the generation history."""
        return self.generation_history
    
    def clear_history(self):
        """Clear the generation history."""
        self.generation_history = []
    
    def save_history(self, filepath: str):
        """Save generation history to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.generation_history, f, indent=2)
        print(f"History saved to {filepath}")


# ============================================================================
# Example usage and demonstrations
# ============================================================================

def example_basic_generation():
    """Example: Basic text generation."""
    print("=" * 70)
    print("Example 1: Basic Generation")
    print("=" * 70)
    
    runner = OllamaLlamaRunner()
    
    prompt = "Explain the concept of entropy in information theory in simple terms."
    
    result = runner.generate(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    
    print(f"\nPrompt: {result['prompt']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nTokens generated: {result['eval_count']}")
    print(f"Time taken: {result['total_duration']:.2f}s")


def example_streaming_generation():
    """Example: Streaming generation with live output."""
    print("\n" + "=" * 70)
    print("Example 2: Streaming Generation")
    print("=" * 70)
    
    runner = OllamaLlamaRunner()
    
    prompt = "Write a short story about a robot learning about information theory."
    
    print(f"\nPrompt: {prompt}")
    print("\nStreaming response:")
    print("-" * 70)
    
    result = runner.generate_streaming(
        prompt=prompt,
        max_tokens=150,
        temperature=0.8,
        verbose=True
    )
    
    print("-" * 70)
    print(f"Tokens generated: {result['tokens_generated']}")


def example_intervention():
    """Example: Generation with keyword-based intervention."""
    print("\n" + "=" * 70)
    print("Example 3: Generation with Intervention")
    print("=" * 70)
    
    runner = OllamaLlamaRunner()
    
    prompt = "Explain machine learning and neural networks."
    keywords = ["deep learning", "backpropagation"]
    
    print(f"\nPrompt: {prompt}")
    print(f"Watching for keywords: {keywords}")
    print("\nResponse:")
    print("-" * 70)
    
    result = runner.intervene_on_keyword(
        prompt=prompt,
        keywords=keywords,
        action="alert",  # Change to "stop" to halt on keyword
        max_tokens=200,
        temperature=0.7,
        verbose=True
    )
    
    print("-" * 70)
    if result['interventions']:
        print(f"\nInterventions triggered: {len(result['interventions'])}")
        for i, intervention in enumerate(result['interventions'], 1):
            print(f"  {i}. Keyword: '{intervention['keyword']}' at position {intervention['position']}")


def example_statistics():
    """Example: Computing statistics on generated text."""
    print("\n" + "=" * 70)
    print("Example 4: Text Statistics")
    print("=" * 70)
    
    runner = OllamaLlamaRunner()
    
    result = runner.generate(
        prompt="Describe the relationship between entropy and information.",
        max_tokens=100,
        temperature=0.7
    )
    
    stats = runner.compute_response_statistics(result['response'])
    
    print(f"\nGenerated text:\n{result['response']}")
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Llama 3.3 via Ollama - Information Theory Integration")
    print("=" * 70)
    
    # Check if ollama is available
    try:
        import ollama
    except ImportError:
        print("\n❌ Error: ollama package not installed")
        print("Install with: pip install ollama")
        return
    
    # Run examples
    try:
        example_basic_generation()
        example_streaming_generation()
        example_intervention()
        example_statistics()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Modify the intervention_callback to implement custom logic")
        print("2. Extend compute_response_statistics() for entropy computation")
        print("3. Integrate with src/information_theory.py for detailed analysis")
        print("4. Use ProbabilityExtractor if you need token-level probabilities")
        print("   (Note: May require raw API access or model modifications)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        print(f"And that llama3.3 is available: ollama pull llama3.3")


if __name__ == "__main__":
    main()
