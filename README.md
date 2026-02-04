# LLM Information Theory Analysis Toolkit

A comprehensive Python toolkit for analyzing information-theoretic properties of large language models. This toolkit enables you to extract **full token probability distributions**, compute **Shannon entropy and information** for each token, and analyze language model uncertainty with precision.

## Features

✨ **Full Distribution Access**: Extract complete probability distributions over all ~50k-128k tokens  
📊 **Entropy Analysis**: Compute Shannon entropy at each token position to measure model uncertainty  
💡 **Information/Surprisal**: Calculate how surprising each token is given its context  
🎯 **Ollama Integration**: Work with local Llama models via Ollama server  
🤗 **Transformers Support**: Access true probability distributions using HuggingFace models  
📈 **Visualization**: Interactive Jupyter notebook with plots, tables, and analysis  
🔧 **Flexible Framework**: Modular design for custom information-theoretic experiments  

## What You Can Do

- **Measure model uncertainty** at each token position (entropy)
- **Quantify token surprise** using Shannon information (surprisal)
- **Compute total sequence entropy** using the chain rule decomposition
- **Calculate perplexity** from conditional entropies
- **Compare models** based on their information-theoretic properties
- **Intervene in generation** with custom prompts and token boosting
- **Visualize** how uncertainty evolves across a sequence

## Requirements

- Python 3.8+ (tested with 3.11.5)
- CUDA-capable GPU (recommended) - tested on RTX 3090
- Ollama server (optional, for local Llama models)
- 8GB+ RAM (16GB+ recommended for larger models)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LLM_Information_theory.git
cd LLM_Information_theory
```

### 2. Set Up Python Environment (Using UV - Recommended)

We use [uv](https://github.com/astral-sh/uv) for fast, reliable Python environment management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

**Alternative: Using standard pip/venv**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Optional: Set Up Ollama (for Local Llama Models)

If you want to use Ollama for local LLM inference:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (e.g., Llama 3.1 8B)
ollama pull llama3.1:8b

# Verify Ollama is running
ollama list
```

### 4. Optional: HuggingFace Authentication

For downloading models from HuggingFace (if not using local weights):

```bash
# Login to HuggingFace
huggingface-cli login
```

Or set your token as an environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Quick Start

### Method 1: Interactive Jupyter Notebook (Recommended)

The easiest way to get started is with our comprehensive notebook:

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter and open the demo notebook
jupyter notebook notebooks/entropy_analysis_demo.ipynb
```

The notebook includes:
- 📚 **Theory explanations** with LaTeX formulas
- 💻 **Executable code cells** you can modify and run
- 📊 **Visualizations**: entropy plots, information bar charts, heatmaps
- 📋 **Tables**: token-by-token analysis with probabilities
- 🔍 **Analysis**: identifying most uncertain/surprising tokens

### Method 2: Command-Line Examples

#### Test Setup (Quick Verification)

```bash
# Quick test with Ollama + Llama 3.1 8B
python examples/quick_test.py

# Expected output: Model generates response about AI
```

#### Entropy Analysis with Full Distributions

```bash
# Analyze entropy using full token distributions (GPT-2 demo)
python examples/entropy_full_distribution.py

# This shows:
# - Entropy at each token position (model's uncertainty)
# - Information/surprisal of each actual token (how surprising)
# - Total sequence entropy via chain rule
# - Perplexity computation
```

#### Ollama Integration

```bash
# Run Llama model via Ollama server
python examples/llama_ollama_runner.py

# Uses local llama3.1:8b model
# Demonstrates basic generation and intervention
```

### Method 3: Python API

```python
from examples.entropy_full_distribution import FullDistributionEntropyAnalyzer

# Initialize analyzer with a model
analyzer = FullDistributionEntropyAnalyzer(
    model_path="gpt2",  # or "meta-llama/Llama-3.1-8B"
    device="cuda"
)

# Analyze a string
results = analyzer.analyze_string(
    sample_string="hello what is the entropy of this string",
    prompt=""
)

# Access results
print(f"Total entropy: {results['total_entropy_sum']:.2f} bits")
print(f"Perplexity: {results['perplexity']:.2f}")

# Token-by-token details
for token, entropy, info in zip(
    results['tokens'], 
    results['entropies'], 
    results['information']
):
    print(f"{token}: H={entropy:.2f} bits, I={info:.2f} bits")
```

## Core Components

### 1. Information Theory Library (`src/information_theory.py`)

Core functions for computing information-theoretic measures:

```python
from src.information_theory import (
    compute_entropy,              # Shannon entropy H = -Σ p*log(p)
    compute_shannon_information,  # Surprisal I = -log(P)
    compute_perplexity,          # Perplexity = 2^H
)

# Example: Compute entropy of a probability distribution
probs = [0.7, 0.2, 0.1]
entropy = compute_entropy(probs, base=2.0)  # In bits
print(f"Entropy: {entropy:.4f} bits")

# Example: Compute surprisal of an observed event
prob_observed = 0.05
information = compute_shannon_information(prob_observed, base=2.0)
print(f"Information: {information:.4f} bits")  # Higher = more surprising
```

### 2. Full Distribution Entropy Analyzer (`examples/entropy_full_distribution.py`)

The **key tool** for accessing complete probability distributions:

```python
from examples.entropy_full_distribution import FullDistributionEntropyAnalyzer

analyzer = FullDistributionEntropyAnalyzer(
    model_path="gpt2",
    device="cuda"
)

# Get full distribution over ALL tokens (~50k for GPT-2)
probs, token_prob, entropy, top_alternatives = analyzer.get_full_next_token_distribution(
    context="The capital of France is",
    target_token=" Paris"
)

print(f"Entropy of distribution: {entropy:.2f} bits")
print(f"Probability of ' Paris': {token_prob:.6f}")
print(f"Information content: {-np.log2(token_prob):.2f} bits")
```

**Why this is important:** Unlike Ollama's API which only returns the sampled token, this gives you:
- Complete probability distribution over **all** vocabulary tokens
- Exact entropy computation (not approximations)
- True Shannon information for any token
- Access to all alternative tokens and their probabilities

### 3. Ollama Integration (`examples/llama_ollama_runner.py`)

For working with local Llama models via Ollama:

```python
from examples.llama_ollama_runner import OllamaLlamaRunner

runner = OllamaLlamaRunner(model="llama3.1:8b")

# Basic generation
response = runner.generate(
    prompt="Explain quantum computing",
    max_tokens=100
)

# Generation with keyword-based intervention
response = runner.intervene_on_keyword(
    base_prompt="The future of AI is",
    keyword="uncertain",
    intervention_prompt=" fascinating and",
    max_tokens=50
)
```

**Note:** Ollama provides a simplified API but doesn't expose full probability distributions. Use the transformers-based analyzer for complete entropy analysis.

### 4. Probability Extractor (`src/probability_extractor.py`)

Original component for probability extraction (legacy support):

- `get_next_token_probabilities()`: Get probability distribution for next token
- `generate_with_probabilities()`: Generate text while tracking probabilities
- `get_token_probability()`: Get probability of a specific token

### 5. Intervention Manager (`src/intervention.py`)

Framework for intervening in generation:

- `generate_with_intervention()`: Generate with custom interventions
- `create_token_boost_intervention()`: Boost specific token probabilities
- `create_token_suppression_intervention()`: Suppress specific tokens

## Usage Examples

### Example 1: Compute Entropy for a String (Full Distribution)

The most accurate way to compute entropy - using the complete probability distribution:

```python
from examples.entropy_full_distribution import FullDistributionEntropyAnalyzer

# Initialize analyzer
analyzer = FullDistributionEntropyAnalyzer(model_path="gpt2", device="cuda")

# Analyze a string
sample_string = "The quick brown fox jumps over the lazy dog"
results = analyzer.analyze_string(sample_string, prompt="")

# Results include:
print(f"Total entropy (via chain rule): {results['total_entropy_sum']:.2f} bits")
print(f"Average entropy per token: {results['total_entropy_avg']:.2f} bits")
print(f"Perplexity: {results['perplexity']:.2f}")

# Token-by-token breakdown
for i, (token, ent, info) in enumerate(zip(
    results['tokens'], 
    results['entropies'], 
    results['information']
)):
    print(f"Position {i+1}: '{token}' | Entropy: {ent:.2f} bits | Information: {info:.2f} bits")
```

**Key insight:** The total entropy of the sequence equals the sum of conditional entropies at each position (chain rule).

### Example 2: Understanding Entropy vs Information

```python
# Entropy = uncertainty BEFORE observing the token
# Information = surprise AFTER observing the token

context = "The capital of France is"
target_token = " Paris"

probs, token_prob, entropy, alternatives = analyzer.get_full_next_token_distribution(
    context=context,
    target_token=target_token
)

print(f"Entropy (model's uncertainty): {entropy:.2f} bits")
print(f"  → Model is uncertain among ~{2**entropy:.0f} tokens")

information = -np.log2(token_prob)
print(f"Information of '{target_token}': {information:.2f} bits")
print(f"  → Probability: {token_prob:.6f}")

if information < entropy:
    print("  → Token was LESS surprising than average")
else:
    print("  → Token was MORE surprising than average")
```

### Example 3: Comparing Models

```python
# Compare GPT-2 vs Llama on the same text
models = ["gpt2", "meta-llama/Llama-3.1-8B"]
text = "Machine learning is revolutionizing"

for model_name in models:
    analyzer = FullDistributionEntropyAnalyzer(model_path=model_name)
    results = analyzer.analyze_string(text, prompt="")
    
    print(f"\n{model_name}:")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Avg Entropy: {results['total_entropy_avg']:.2f} bits")
    print(f"  Total Information: {results['total_information']:.2f} bits")
```

Lower perplexity = better model (more confident predictions).

### Example 4: Ollama Integration with Llama

```python
from examples.llama_ollama_runner import OllamaLlamaRunner

runner = OllamaLlamaRunner(model="llama3.1:8b")

# Simple generation
response = runner.generate(
    prompt="Explain the concept of entropy in information theory:",
    max_tokens=150,
    temperature=0.7
)
print(response)

# Streaming generation
for chunk in runner.generate_streaming(
    prompt="Write a short story about",
    max_tokens=100
):
    print(chunk, end='', flush=True)
```

**Note:** Ollama is great for generation but doesn't expose full distributions. Use transformers-based analyzer for entropy computation.

## Project Structure

```
LLM_Information_theory/
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── information_theory.py          # Core: compute_entropy, compute_shannon_information
│   ├── probability_extractor.py       # Token probability extraction (legacy)
│   └── intervention.py                # Generation intervention framework
│
├── examples/
│   ├── entropy_full_distribution.py   # ⭐ PRIMARY: Full distribution entropy analyzer
│   ├── llama_ollama_runner.py         # Ollama integration for local Llama models
│   ├── quick_test.py                  # Fast setup verification
│   ├── entropy_real_probabilities.py  # Real entropy with GPT-2
│   ├── entropy_intervention_example.py # Mock framework demonstration
│   └── README.md                      # Examples documentation
│
├── notebooks/
│   └── entropy_analysis_demo.ipynb    # ⭐ Interactive notebook with plots/tables
│
├── docs/
│   ├── overview.md                    # Project overview
│   └── entropy_information_theory.md  # Comprehensive theory documentation
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

### Key Files

- **`entropy_full_distribution.py`**: The main tool for accurate entropy/information analysis
- **`entropy_analysis_demo.ipynb`**: Best starting point - interactive with visualizations
- **`information_theory.py`**: Core computation functions you'll use everywhere
- **`llama_ollama_runner.py`**: Simple Ollama integration (generation only)
- **`entropy_information_theory.md`**: Deep dive into the theory with worked examples

## Testing Your Setup

### 1. Verify Environment

```bash
# Activate environment
source .venv/bin/activate

# Check Python version
python --version  # Should be 3.8+

# Verify GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Quick Ollama Test

```bash
# Check Ollama is running
ollama list

# Run quick test
python examples/quick_test.py

# Expected output:
# ✓ Successfully connected to Ollama
# ✓ Model llama3.1:8b is available
# Generated response about artificial intelligence
```

### 3. Test Entropy Computation

```bash
# Run full distribution entropy analysis
python examples/entropy_full_distribution.py

# Expected output:
# - Loading model message
# - Token-by-token table with entropy and information
# - Summary statistics
# - Total entropy, perplexity, etc.
```

### 4. Test Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/entropy_analysis_demo.ipynb

# In the notebook:
# 1. Run the import cell (should succeed)
# 2. Run the GPU check cell (verify CUDA if available)
# 3. Run the analyzer initialization cell (loads model)
# 4. Run analysis cells (generates plots and tables)
```

### 5. Expected Results

After running `entropy_full_distribution.py`, you should see output like:

```
================================================================================
TOKEN-BY-TOKEN ANALYSIS
================================================================================
Position    Token    Probability    Entropy (bits)    Information (bits)
       1    hello        0.000234            9.3421               12.0456
       2     what        0.012345            7.5432                6.3421
       3       is        0.087654            6.8765                3.5123
...
================================================================================

================================================================================
SUMMARY STATISTICS
================================================================================
                   Metric                    Value
            Total Tokens                       10
          Average Entropy              8.2345 bits
  Total Entropy (Sequence)            82.3450 bits
                      ...                      ...
```

### Troubleshooting Tests

**Import Error: No module named 'transformers'**
```bash
uv pip install transformers torch
```

**Ollama connection error**
```bash
# Start Ollama service
ollama serve

# In another terminal, test
ollama run llama3.1:8b
```

**CUDA out of memory**
- Use smaller model: `model_path="gpt2"` instead of Llama
- Or run on CPU: `device="cpu"` (slower but works)

**NaN entropy values**
- This is fixed in current version (filters zero probabilities)
- Make sure you have latest code

## Theory Background

### Shannon Entropy

Measures **uncertainty** in a probability distribution:

$$H(P) = -\sum_{i=1}^{V} P(x_i) \log_2 P(x_i)$$

- Higher entropy = more uncertain (flat distribution)
- Lower entropy = more certain (peaked distribution)
- Units: **bits** (with log base 2)

### Shannon Information (Surprisal)

Measures how **surprising** an observed event is:

$$I(x) = -\log_2 P(x)$$

- Higher information = more surprising (low probability)
- Lower information = less surprising (high probability)
- Example: P=0.5 → I=1 bit, P=0.25 → I=2 bits

### Chain Rule for Sequences

The total entropy of a sequence decomposes as:

$$H(T_1, T_2, ..., T_n) = \sum_{i=1}^{n} H(T_i | T_{1:i-1})$$

This is exactly what we compute: the sum of conditional entropies at each position.

### Perplexity

Average branching factor - how many tokens the model is "choosing" from:

$$\text{Perplexity} = 2^{H}$$

Lower perplexity = better model (more confident/accurate predictions).

**For more details**, see [`docs/entropy_information_theory.md`](docs/entropy_information_theory.md) which includes:
- Complete mathematical derivations
- Worked examples with actual numbers
- Relationship between entropy, information, and perplexity
- Why mock probabilities give constant entropy
- How to interpret results

## Key Insights

### Why Full Distributions Matter

**Ollama API limitation:**
- Only returns sampled tokens
- Cannot compute true entropy (need all probabilities)
- Good for generation, not for information theory analysis

**Transformers approach (our solution):**
- Access to full logits over all ~50k-128k tokens
- Apply softmax to get complete probability distribution
- Compute exact entropy: $H = -\sum_{i=1}^{V} p_i \log_2 p_i$
- Get true probability of any token

### Mock vs Real Probabilities

**Mock probabilities** (in `entropy_intervention_example.py`):
- Use same distribution shape for all positions
- Result: constant entropy (~3.61 bits)
- Good for: testing framework, not real analysis

**Real probabilities** (in `entropy_full_distribution.py`):
- Extract from actual language model
- Entropy varies widely (5-11 bits typically)
- Shows true model uncertainty at each position

### Entropy vs Information

| Metric | Measures | When | Formula |
|--------|----------|------|---------|
| **Entropy** | Uncertainty | BEFORE observing token | $H = -\sum p_i \log p_i$ |
| **Information** | Surprise | AFTER observing token | $I = -\log P(x)$ |

- **High entropy + low information**: Model uncertain, but got an expected token
- **Low entropy + high information**: Model confident, but token was surprising
- **Average information ≈ average entropy**: Typical outcome
- **Information > entropy**: More surprising than typical

## Advanced Usage

### Custom Analysis

Create your own analyzers by extending the base class:

```python
from examples.entropy_full_distribution import FullDistributionEntropyAnalyzer
import numpy as np

class CustomAnalyzer(FullDistributionEntropyAnalyzer):
    def analyze_with_alternatives(self, text, top_k=5):
        """Show top-k alternatives at each position."""
        results = self.analyze_string(text, prompt="")
        
        for i, (token, entropy) in enumerate(zip(results['tokens'], results['entropies'])):
            print(f"\nPosition {i+1}: '{token}' (H={entropy:.2f} bits)")
            
            # Get alternatives at this position
            context = ' '.join(results['tokens'][:i])
            _, _, _, alternatives = self.get_full_next_token_distribution(
                context=context, 
                target_token=token
            )
            
            print("  Top alternatives:")
            for alt_token, alt_prob in alternatives[:top_k]:
                alt_info = -np.log2(alt_prob)
                print(f"    '{alt_token}': P={alt_prob:.6f} (I={alt_info:.2f} bits)")
```

### Intervention Experiments

Combine Ollama generation with entropy monitoring:

```python
from examples.llama_ollama_runner import OllamaLlamaRunner
from examples.entropy_full_distribution import FullDistributionEntropyAnalyzer

# Generate with Ollama
runner = OllamaLlamaRunner(model="llama3.1:8b")
text = runner.generate("The future of AI", max_tokens=50)

# Analyze entropy with transformers
analyzer = FullDistributionEntropyAnalyzer("gpt2")
results = analyzer.analyze_string(text, prompt="")

print(f"Generated text perplexity: {results['perplexity']:.2f}")
print(f"Average entropy: {results['total_entropy_avg']:.2f} bits")
```

### Performance and Memory

**GPU Acceleration:**
```python
# Use GPU for 10-100x speedup
analyzer = FullDistributionEntropyAnalyzer(
    model_path="gpt2",
    device="cuda"  # Automatically uses CUDA if available
)

# Verify GPU usage
import torch
print(f"Using device: {analyzer.device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Memory Optimization:**
- Use smaller models: `"gpt2"` (~500MB) vs `"meta-llama/Llama-3.1-8B"` (~8GB)
- CPU-only mode: `device="cpu"` (slower but no GPU memory needed)
- Batch analysis: Analyze multiple strings without reloading model

**Speed Tips:**
1. GPU is essential for real-time analysis (10-100x faster)
2. Model size tradeoff: GPT-2 (small, fast) vs Llama (large, accurate)
3. Models are cached by HuggingFace after first download

## Troubleshooting

### Installation Issues

**Import Error: No module named 'transformers'**
```bash
source .venv/bin/activate
uv pip install transformers torch
```

**UV not found**
```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then restart your terminal
```

### Ollama Issues

**Ollama connection error**
```bash
# Start Ollama service
ollama serve

# In another terminal, verify
ollama list
ollama run llama3.1:8b
```

**Model not found**
```bash
# Pull the model first
ollama pull llama3.1:8b

# List available models
ollama list
```

### GPU/Memory Issues

**CUDA out of memory**
- Use smaller model: `model_path="gpt2"` instead of Llama
- Run on CPU: `device="cpu"` (slower but works)
- Close other GPU applications
- Reduce sequence length

**CUDA not available but GPU exists**
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# May need to reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

**NaN entropy values**
- This is fixed in current version (filters zero probabilities)
- Update to latest code with `git pull`

**Slow generation**
- Verify GPU usage: `torch.cuda.is_available()`
- Use smaller models for experimentation
- First run is always slower (model download/loading)

**Model access/authentication issues**
- Login to HuggingFace: `huggingface-cli login`
- Check model access permissions
- Some models require approval (Llama models)

### Notebook Issues

**Jupyter not found**
```bash
source .venv/bin/activate
uv pip install jupyter
```

**Kernel dies when loading model**
- Model too large for available memory
- Use `"gpt2"` for testing
- Or use CPU mode: `device="cpu"`

## FAQ

**Q: Why use transformers instead of Ollama for entropy?**  
A: Ollama's API doesn't expose full probability distributions over all tokens - only the sampled token. To compute true entropy ($H = -\sum p_i \log p_i$), you need probabilities for ALL tokens, which requires accessing the model's logits directly via transformers.

**Q: Can I use this with Llama 3.1 or 3.3?**  
A: Yes! Use `model_path="meta-llama/Llama-3.1-8B"` or point to local weights. You may need HuggingFace authentication and model access approval.

**Q: What's the difference between entropy and information?**  
A: **Entropy** measures uncertainty in the distribution BEFORE observing a token. **Information** (surprisal) measures how surprising the actual observed token was. Entropy is computed over all possible tokens; information is computed for one specific token.

**Q: Why is total entropy = sum of entropies?**  
A: This is the chain rule! $H(T_1, T_2, ..., T_n) = \sum_i H(T_i | T_{1:i-1})$. Each entropy we compute is conditional on previous tokens, so summing gives the joint entropy of the entire sequence.

**Q: What's a "good" perplexity value?**  
A: Lower is better. GPT-2 on typical text: ~20-50. State-of-the-art models: ~10-20. Very predictable text: <10. Random/incoherent text: >100.

**Q: Can I analyze text generated by Ollama?**  
A: Yes! Generate text with Ollama, then analyze it with the transformers-based entropy analyzer. They don't need to use the same model.

## Out of Memory Error

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
  author={Stuart Synakowski},
  year={2026},
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

- [x] Full probability distribution access via transformers
- [x] Entropy and information computation with chain rule
- [x] Interactive Jupyter notebook with visualizations
- [x] Ollama integration for local Llama models
- [x] Comprehensive theory documentation
- [ ] Real-time intervention during generation
- [ ] Multi-model comparison tools
- [ ] Varentropy analysis (entropy variance over time)
- [ ] KL divergence between model distributions
- [ ] Export results to CSV/JSON
- [ ] Streaming analysis for long texts
- [ ] Multi-GPU support for large models
- [ ] Web interface for interactive exploration

## References

- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory"
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Ollama: https://ollama.com/

---

**Happy experimenting with LLM information theory!** 🚀

For questions or contributions, please open an issue or PR on GitHub.
