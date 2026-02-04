"""
Quick test script for Ollama Llama runner
Uses the smaller llama3.1:8b model for faster testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_ollama_runner import OllamaLlamaRunner

def main():
    print("=" * 70)
    print("Quick Test: Llama via Ollama")
    print("=" * 70)
    
    # Use the smaller model for faster testing
    runner = OllamaLlamaRunner(model_name="llama3.1:8b")
    
    # Short prompt with limited tokens for quick test
    prompt = "What is 2+2?"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    
    result = runner.generate(
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    
    print(f"\nResponse: {result['response']}")
    print(f"\nStats:")
    print(f"  - Tokens generated: {result['eval_count']}")
    print(f"  - Time: {result['total_duration']:.2f}s")
    
    # Test streaming
    print("\n" + "=" * 70)
    print("Testing streaming generation...")
    print("=" * 70)
    
    prompt2 = "Count from 1 to 5."
    print(f"\nPrompt: {prompt2}")
    print("Response: ", end="", flush=True)
    
    result2 = runner.generate_streaming(
        prompt=prompt2,
        max_tokens=30,
        temperature=0.7,
        verbose=True
    )
    
    print(f"\n✓ Test completed successfully!")

if __name__ == "__main__":
    main()
