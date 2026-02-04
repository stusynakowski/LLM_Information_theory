# LLM Information Theory Analysis Toolkit

A comprehensive Python toolkit for analyzing information-theoretic properties of large language models, with a focus on local Llama models. This toolkit enables you to extract token probabilities, intervene in the generation process, and compute entropy and Shannon information for each generated token.

## Features

✨ **Probability Extraction**: Access decoder probabilities for each output token  
🎯 **Generation Intervention**: Modify the generation process in real-time  
📊 **Information Theory**: Compute entropy, Shannon information, perplexity, and more  
📈 **Visualization**: Built-in plotting for probability distributions and surprisal  
🔧 **Flexible**: Works with any HuggingFace-compatible decoder model  

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for larger models)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LLM_Information_theory.git
cd LLM_Information_theory
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Access

For Llama models, you'll need:
- A HuggingFace account
- Access to the Llama model repository
- Authentication token

```bash
# Login to HuggingFace
huggingface-cli login
```

Or set your token as an environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Quick Start

### Using the Jupyter Notebook (Recommended)

1. Start Jupyter:
```bash
jupyter notebook notebooks/information_theory_demo.ipynb
```

2. Follow the interactive demonstrations in the notebook

### Python Script Example

```python
from src.probability_extractor import ProbabilityExtractor
from src.information_theory import compute_entropy, compute_shannon_information
from src.intervention import InterventionManager

# Load your model
extractor = ProbabilityExtractor(
    "meta-llama/Llama-2-7b-hf",  # or path to local model
    load_in_8bit=True  # for lower memory usage
)

# Get next token probabilities
result = extractor.get_next_token_probabilities("The capital of France is")
print(f"Top prediction: {result['top_k_tokens'][0]}")

# Generate with probability tracking
gen_result = extractor.generate_with_probabilities(
    prompt="Artificial intelligence is",
    max_new_tokens=20,
    temperature=0.8
)

# Compute information theory metrics
for token, prob in zip(gen_result['generated_tokens'], gen_result['token_probabilities']):
    surprisal = compute_shannon_information(prob)
    print(f"Token: '{token}' | Probability: {prob:.4f} | Surprisal: {surprisal:.2f} bits")
```

## Core Components

### 1. Probability Extractor (`src/probability_extractor.py`)

Extract and analyze token probabilities from language models:

- `get_next_token_probabilities()`: Get probability distribution for next token
- `generate_with_probabilities()`: Generate text while tracking all probabilities
- `get_token_probability()`: Get probability of a specific token
- `compare_token_probabilities()`: Compare probabilities of multiple tokens

### 2. Information Theory (`src/information_theory.py`)

Compute information-theoretic measures:

- `compute_entropy()`: Shannon entropy of a distribution
- `compute_shannon_information()`: Surprisal of an observed token
- `compute_perplexity()`: Model perplexity
- `compute_kl_divergence()`: KL divergence between distributions
- `compute_varentropy()`: Variance of surprisal
- `analyze_token_information()`: Comprehensive analysis of token sequence

### 3. Intervention Manager (`src/intervention.py`)

Intervene in the generation process:

- `generate_with_intervention()`: Generate with custom interventions
- `create_token_boost_intervention()`: Boost specific token probabilities
- `create_token_suppression_intervention()`: Suppress specific tokens
- `create_entropy_maximization_intervention()`: Target specific entropy levels
- `create_conditional_intervention()`: Apply interventions conditionally
- `analyze_intervention_effects()`: Compare intervention vs baseline

## Usage Examples

### Example 1: Extract Token Probabilities

```python
extractor = ProbabilityExtractor("meta-llama/Llama-2-7b-hf")

# Get next token distribution
result = extractor.get_next_token_probabilities("The meaning of life is", return_top_k=10)

for token, prob in result['top_k_tokens']:
    print(f"{token}: {prob:.4f}")

# Calculate entropy
entropy = compute_entropy(result['probabilities'])
print(f"Distribution entropy: {entropy:.2f} bits")
```

### Example 2: Intervention - Force Specific Tokens

```python
intervention_manager = InterventionManager(extractor)

# Force specific tokens at certain positions
result = intervention_manager.generate_with_intervention(
    prompt="My favorite color is",
    forced_tokens={0: " blue", 2: " because"},
    max_new_tokens=15,
    track_alternatives=True
)

print(result['generated_text'])
```

### Example 3: Boost Certain Topics

```python
# Create intervention to boost science-related words
science_boost = intervention_manager.create_token_boost_intervention(
    boost_tokens=[' science', ' research', ' theory', ' experiment'],
    boost_factor=3.0
)

result = intervention_manager.generate_with_intervention(
    prompt="The discovery was remarkable because",
    intervention_fn=science_boost,
    max_new_tokens=25
)
```

### Example 4: Information Theory Analysis

```python
from src.information_theory import analyze_token_information

result = extractor.generate_with_probabilities(
    prompt="Machine learning models",
    max_new_tokens=30
)

# Comprehensive analysis
info = analyze_token_information(result['token_probabilities'])

print(f"Mean surprisal: {info['mean_surprisal']:.2f} bits")
print(f"Perplexity: {info['perplexity']:.2f}")
print(f"Total information: {info['total_information']:.2f} bits")
```

## Project Structure

```
LLM_Information_theory/
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── probability_extractor.py    # Token probability extraction
│   ├── information_theory.py       # Information theory computations
│   └── intervention.py              # Generation intervention tools
├── notebooks/
│   └── information_theory_demo.ipynb  # Interactive demonstration
├── examples/
│   └── (additional example scripts)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Advanced Features

### Custom Intervention Functions

Create your own intervention strategies:

```python
def custom_intervention(logits, position, context):
    """Apply temperature scaling based on position."""
    temperature = 1.0 + (position * 0.1)  # Increase temp over time
    return logits / temperature

result = intervention_manager.generate_with_intervention(
    prompt="Once upon a time",
    intervention_fn=custom_intervention,
    max_new_tokens=30
)
```

### Conditional Interventions

Apply different interventions based on conditions:

```python
def is_early_generation(position, context):
    return position < 5

conditional = intervention_manager.create_conditional_intervention(
    condition_fn=is_early_generation,
    true_intervention=boost_function,
    false_intervention=suppress_function
)
```

### Comparative Analysis

Analyze the effects of interventions:

```python
analysis = intervention_manager.analyze_intervention_effects(
    prompt="The future of AI",
    intervention_fn=your_intervention,
    num_samples=5
)

print(f"Baseline perplexity: {analysis['mean_baseline_perplexity']:.2f}")
print(f"Intervention perplexity: {analysis['mean_intervention_perplexity']:.2f}")
```

## Memory Optimization

For systems with limited GPU memory:

```python
# 8-bit quantization
extractor = ProbabilityExtractor(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True
)

# 4-bit quantization (even lower memory)
extractor = ProbabilityExtractor(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True
)

# CPU-only mode
extractor = ProbabilityExtractor(
    "meta-llama/Llama-2-7b-hf",
    device="cpu"
)
```

## Performance Tips

1. **Use GPU**: CUDA-enabled GPU provides 10-100x speedup
2. **Batch Processing**: Process multiple prompts together when possible
3. **Quantization**: Use 8-bit or 4-bit models to reduce memory usage
4. **Cache Models**: Models are cached by HuggingFace after first download

## Troubleshooting

### Out of Memory Error

- Use quantization: `load_in_8bit=True` or `load_in_4bit=True`
- Use a smaller model variant
- Reduce `max_new_tokens`
- Close other GPU applications

### Slow Generation

- Ensure you're using GPU: check with `torch.cuda.is_available()`
- Use smaller models for experimentation
- Consider using quantized versions

### Model Access Issues

- Ensure you're logged into HuggingFace: `huggingface-cli login`
- Check you have access to the model repository
- Verify your token has the correct permissions

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{llm_information_theory,
  title={LLM Information Theory Analysis Toolkit},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/LLM_Information_theory}
}
```

## Acknowledgments

- Built on HuggingFace Transformers
- Inspired by information theory research in NLP
- Thanks to the open-source community

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Consult the notebook for usage examples

## Roadmap

- [ ] Support for encoder-decoder models
- [ ] Multi-GPU support
- [ ] Streaming generation with interventions
- [ ] Export results to various formats
- [ ] Pre-built intervention strategies library
- [ ] Integration with popular LLM frameworks

---

**Happy experimenting with LLM information theory!** 🚀
