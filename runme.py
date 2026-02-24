from examples.llama_ollama_runner import OllamaLlamaRunner

runner = OllamaLlamaRunner()

# Generate with intervention callback
def entropy_callback(full_text, new_chunk):
    # Your custom logic here - compute entropy, etc.
    if some_condition:
        return "STOP"
    return None

result = runner.generate_streaming(
    prompt="Your prompt",
    intervention_callback=entropy_callback
)