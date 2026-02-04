"""
Probability Extractor Module

This module provides utilities to extract token probabilities from language models,
specifically designed for decoder-only models like Llama.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class ProbabilityExtractor:
    """
    Extracts token probabilities from language model generations.
    
    This class wraps a HuggingFace language model and provides methods to:
    - Generate text while tracking probabilities for each token
    - Access logits and probability distributions
    - Retrieve top-k most likely tokens at each position
    """
    
    def __init__(
        self, 
        model_name: str, 
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the probability extractor with a language model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device to load model on ("cuda", "cpu", or None for auto)
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model: {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        model_kwargs = {}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def get_next_token_probabilities(
        self, 
        prompt: str,
        return_top_k: int = 10
    ) -> Dict:
        """
        Get probability distribution for the next token given a prompt.
        
        Args:
            prompt: Input text prompt
            return_top_k: Number of top probable tokens to return
            
        Returns:
            Dictionary containing:
                - 'prompt': Original prompt
                - 'logits': Raw logits for all tokens
                - 'probabilities': Probability distribution over all tokens
                - 'top_k_tokens': List of (token, probability) tuples for top-k tokens
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for next token
            probs = F.softmax(logits, dim=-1)
        
        # Get top-k tokens
        top_probs, top_indices = torch.topk(probs, return_top_k)
        top_tokens = [
            (self.tokenizer.decode([idx]), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return {
            'prompt': prompt,
            'logits': logits.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'top_k_tokens': top_tokens
        }
    
    def generate_with_probabilities(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        return_full_distribution: bool = False
    ) -> Dict:
        """
        Generate text while tracking probabilities for each generated token.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            return_full_distribution: Whether to return full probability distribution
            
        Returns:
            Dictionary containing:
                - 'prompt': Original prompt
                - 'generated_text': Full generated text
                - 'generated_tokens': List of generated tokens
                - 'token_probabilities': Probability of each generated token
                - 'token_logits': Logits for each generated token (if return_full_distribution)
                - 'full_distributions': Full probability distributions (if return_full_distribution)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        generated_tokens = []
        token_probabilities = []
        token_logits = [] if return_full_distribution else None
        full_distributions = [] if return_full_distribution else None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model outputs
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering if specified
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Store information
                token_str = self.tokenizer.decode([next_token.item()])
                generated_tokens.append(token_str)
                token_probabilities.append(probs[next_token].item())
                
                if return_full_distribution:
                    token_logits.append(next_token_logits.cpu().numpy())
                    full_distributions.append(probs.cpu().numpy())
                
                # Append to input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.tokenizer.decode(input_ids[0])
        
        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'generated_tokens': generated_tokens,
            'token_probabilities': token_probabilities,
        }
        
        if return_full_distribution:
            result['token_logits'] = token_logits
            result['full_distributions'] = full_distributions
        
        return result
    
    def get_token_probability(
        self,
        prompt: str,
        target_token: str
    ) -> float:
        """
        Get the probability of a specific target token given a prompt.
        
        Args:
            prompt: Input text prompt
            target_token: The token to get probability for
            
        Returns:
            Probability of the target token
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
        
        return probs[target_id].item()
    
    def compare_token_probabilities(
        self,
        prompt: str,
        tokens: List[str]
    ) -> Dict[str, float]:
        """
        Compare probabilities of multiple tokens given a prompt.
        
        Args:
            prompt: Input text prompt
            tokens: List of tokens to compare
            
        Returns:
            Dictionary mapping each token to its probability
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
        
        token_probs = {}
        for token in tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 0:
                token_probs[token] = probs[token_ids[0]].item()
            else:
                token_probs[token] = 0.0
        
        return token_probs
