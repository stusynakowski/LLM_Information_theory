# Examples

This directory contains example scripts demonstrating how to use the LLM Information Theory toolkit.

## Llama 3.3 via Ollama

The `llama_ollama_runner.py` script provides a template for running Llama 3.3 through Ollama with support for information-theoretic analysis.

### Prerequisites

1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai)
2. **Pull Llama 3.3**: `ollama pull llama3.3`
3. **Install Python package**: `pip install ollama`

### Usage

Run all examples:
```bash
python examples/llama_ollama_runner.py
```

Or import and use in your own scripts:
```python
from examples.llama_ollama_runner import OllamaLlamaRunner

runner = OllamaLlamaRunner(model_name="llama3.3")

# Basic generation
result = runner.generate("Your prompt here", max_tokens=200)
print(result['response'])

# Streaming with intervention
def my_callback(full_text, new_chunk):
    if "stop_word" in full_text:
        return "STOP"
    return None

result = runner.generate_streaming(
    prompt="Your prompt here",
    intervention_callback=my_callback
)
```

### Features

- **Basic Generation**: Simple text generation with configurable parameters
- **Streaming Generation**: Real-time token-by-token generation with callbacks
- **Intervention**: Stop or alert on keywords during generation
- **Statistics**: Compute basic text statistics (extensible for entropy)
- **History Tracking**: Keep track of all generations

### Extending for Information Theory

The script is designed to be extended with:

1. **Entropy Computation**: Add token probability extraction to compute entropy
2. **Surprisal Analysis**: Integrate with `src/information_theory.py`
3. **Intervention Logic**: Modify probability distributions during generation
4. **Custom Metrics**: Add domain-specific information measures

Note: Ollama's standard API doesn't expose token probabilities by default. For detailed probability analysis, you may need to use the raw API or integrate with the model directly.
