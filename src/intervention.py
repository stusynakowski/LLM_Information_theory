"""
Intervention Module

This module provides utilities to intervene in language model generation,
allowing you to modify probabilities, force specific tokens, or adjust
the generation process at each step.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Dict, List, Tuple, Union
import numpy as np


class InterventionManager:
    """
    Manages interventions during language model generation.
    
    This class allows you to:
    - Modify probability distributions before sampling
    - Force specific tokens at certain positions
    - Apply custom transformations to logits
    - Track intervention effects
    """
    
    def __init__(self, probability_extractor):
        """
        Initialize the intervention manager.
        
        Args:
            probability_extractor: An instance of ProbabilityExtractor
        """
        self.extractor = probability_extractor
        self.model = probability_extractor.model
        self.tokenizer = probability_extractor.tokenizer
        self.device = probability_extractor.device
        
        # Track interventions
        self.intervention_history = []
    
    def generate_with_intervention(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        intervention_fn: Optional[Callable] = None,
        intervention_positions: Optional[List[int]] = None,
        forced_tokens: Optional[Dict[int, str]] = None,
        temperature: float = 1.0,
        track_alternatives: bool = True
    ) -> Dict:
        """
        Generate text with interventions at specified positions.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            intervention_fn: Custom function to modify logits/probs at each step
                           Signature: fn(logits, position, context) -> modified_logits
            intervention_positions: List of positions where intervention_fn should be applied
            forced_tokens: Dict mapping position -> token to force at that position
            temperature: Sampling temperature
            track_alternatives: Whether to track what would have been generated without intervention
            
        Returns:
            Dictionary containing generation results and intervention effects
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        generated_tokens = []
        token_probabilities = []
        intervention_log = []
        alternatives = [] if track_alternatives else None
        
        with torch.no_grad():
            for position in range(max_new_tokens):
                # Get model outputs
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :] / temperature
                
                # Store original distribution if tracking alternatives
                if track_alternatives:
                    original_probs = F.softmax(next_token_logits.clone(), dim=-1)
                    original_top_token = torch.argmax(original_probs)
                    original_top_prob = original_probs[original_top_token].item()
                
                # Apply forced token if specified for this position
                intervention_applied = False
                if forced_tokens and position in forced_tokens:
                    forced_token = forced_tokens[position]
                    forced_token_id = self.tokenizer.encode(forced_token, add_special_tokens=False)[0]
                    next_token = torch.tensor([forced_token_id], device=self.device)
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    token_prob = probs[forced_token_id].item()
                    
                    intervention_applied = True
                    intervention_type = "forced_token"
                    intervention_details = {
                        'forced_token': forced_token,
                        'original_probability': token_prob
                    }
                
                # Apply custom intervention function
                elif intervention_fn and (intervention_positions is None or position in intervention_positions):
                    context = {
                        'position': position,
                        'prompt': prompt,
                        'generated_so_far': self.tokenizer.decode(input_ids[0]),
                        'input_ids': input_ids
                    }
                    
                    modified_logits = intervention_fn(next_token_logits, position, context)
                    probs = F.softmax(modified_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_prob = probs[next_token].item()
                    
                    intervention_applied = True
                    intervention_type = "custom_function"
                    intervention_details = {
                        'function': intervention_fn.__name__ if hasattr(intervention_fn, '__name__') else 'lambda'
                    }
                
                # No intervention - standard sampling
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_prob = probs[next_token].item()
                    intervention_type = "none"
                    intervention_details = {}
                
                # Store token information
                token_str = self.tokenizer.decode([next_token.item()])
                generated_tokens.append(token_str)
                token_probabilities.append(token_prob)
                
                # Log intervention
                intervention_entry = {
                    'position': position,
                    'token': token_str,
                    'probability': token_prob,
                    'intervention': intervention_applied,
                    'intervention_type': intervention_type,
                    'details': intervention_details
                }
                
                if track_alternatives and intervention_applied:
                    intervention_entry['alternative_token'] = self.tokenizer.decode([original_top_token.item()])
                    intervention_entry['alternative_probability'] = original_top_prob
                
                intervention_log.append(intervention_entry)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.tokenizer.decode(input_ids[0])
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'generated_tokens': generated_tokens,
            'token_probabilities': token_probabilities,
            'intervention_log': intervention_log
        }
    
    def create_token_boost_intervention(
        self,
        boost_tokens: List[str],
        boost_factor: float = 2.0
    ) -> Callable:
        """
        Create an intervention function that boosts probabilities of specific tokens.
        
        Args:
            boost_tokens: List of tokens to boost
            boost_factor: Multiplicative factor to apply to logits
            
        Returns:
            Intervention function
        """
        boost_token_ids = []
        for token in boost_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            boost_token_ids.extend(token_ids)
        
        def boost_intervention(logits, position, context):
            modified_logits = logits.clone()
            for token_id in boost_token_ids:
                modified_logits[token_id] *= boost_factor
            return modified_logits
        
        boost_intervention.__name__ = f"boost_{len(boost_tokens)}_tokens"
        return boost_intervention
    
    def create_token_suppression_intervention(
        self,
        suppress_tokens: List[str],
        suppression_strength: float = 0.01
    ) -> Callable:
        """
        Create an intervention function that suppresses probabilities of specific tokens.
        
        Args:
            suppress_tokens: List of tokens to suppress
            suppression_strength: Multiplicative factor (< 1.0) to apply to logits
            
        Returns:
            Intervention function
        """
        suppress_token_ids = []
        for token in suppress_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            suppress_token_ids.extend(token_ids)
        
        def suppress_intervention(logits, position, context):
            modified_logits = logits.clone()
            for token_id in suppress_token_ids:
                modified_logits[token_id] *= suppression_strength
            return modified_logits
        
        suppress_intervention.__name__ = f"suppress_{len(suppress_tokens)}_tokens"
        return suppress_intervention
    
    def create_entropy_maximization_intervention(
        self,
        target_entropy: float,
        learning_rate: float = 0.1
    ) -> Callable:
        """
        Create an intervention that adjusts temperature to target a specific entropy.
        
        Args:
            target_entropy: Desired entropy in bits
            learning_rate: How aggressively to adjust temperature
            
        Returns:
            Intervention function
        """
        def entropy_intervention(logits, position, context):
            # Try different temperatures to match target entropy
            best_temp = 1.0
            best_diff = float('inf')
            
            for temp in np.linspace(0.1, 3.0, 20):
                scaled_logits = logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()
                diff = abs(entropy - target_entropy)
                
                if diff < best_diff:
                    best_diff = diff
                    best_temp = temp
            
            return logits / best_temp
        
        entropy_intervention.__name__ = f"target_entropy_{target_entropy}"
        return entropy_intervention
    
    def create_conditional_intervention(
        self,
        condition_fn: Callable,
        true_intervention: Callable,
        false_intervention: Optional[Callable] = None
    ) -> Callable:
        """
        Create an intervention that applies different functions based on a condition.
        
        Args:
            condition_fn: Function that returns True/False based on context
            true_intervention: Intervention to apply if condition is True
            false_intervention: Intervention to apply if condition is False (or None for no intervention)
            
        Returns:
            Conditional intervention function
        """
        def conditional_intervention(logits, position, context):
            if condition_fn(position, context):
                return true_intervention(logits, position, context)
            elif false_intervention:
                return false_intervention(logits, position, context)
            else:
                return logits
        
        conditional_intervention.__name__ = "conditional_intervention"
        return conditional_intervention
    
    def analyze_intervention_effects(
        self,
        prompt: str,
        intervention_fn: Callable,
        max_new_tokens: int = 30,
        num_samples: int = 5
    ) -> Dict:
        """
        Analyze the effects of an intervention by comparing with baseline generation.
        
        Args:
            prompt: Input prompt
            intervention_fn: Intervention function to analyze
            max_new_tokens: Number of tokens to generate
            num_samples: Number of samples to generate for comparison
            
        Returns:
            Analysis results comparing intervention vs baseline
        """
        from .information_theory import analyze_token_information
        
        # Generate with intervention
        intervention_results = []
        for _ in range(num_samples):
            result = self.generate_with_intervention(
                prompt,
                max_new_tokens=max_new_tokens,
                intervention_fn=intervention_fn,
                track_alternatives=True
            )
            intervention_results.append(result)
        
        # Generate baseline (no intervention)
        baseline_results = []
        for _ in range(num_samples):
            result = self.generate_with_intervention(
                prompt,
                max_new_tokens=max_new_tokens,
                intervention_fn=None,
                track_alternatives=False
            )
            baseline_results.append(result)
        
        # Analyze information content
        intervention_info = [
            analyze_token_information(r['token_probabilities'])
            for r in intervention_results
        ]
        baseline_info = [
            analyze_token_information(r['token_probabilities'])
            for r in baseline_results
        ]
        
        return {
            'intervention_samples': intervention_results,
            'baseline_samples': baseline_results,
            'intervention_info': intervention_info,
            'baseline_info': baseline_info,
            'mean_intervention_perplexity': np.mean([i['perplexity'] for i in intervention_info]),
            'mean_baseline_perplexity': np.mean([i['perplexity'] for i in baseline_info]),
            'mean_intervention_entropy': np.mean([i['mean_surprisal'] for i in intervention_info]),
            'mean_baseline_entropy': np.mean([i['mean_surprisal'] for i in baseline_info])
        }
