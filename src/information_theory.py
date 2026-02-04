"""
Information Theory Module

This module provides functions to compute information-theoretic measures
for language model token generation, including entropy and Shannon information.
"""

import numpy as np
from typing import Union, List, Dict, Optional
import torch


def compute_entropy(
    probabilities: Union[np.ndarray, torch.Tensor, List[float]],
    base: float = 2.0,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the Shannon entropy of a probability distribution.
    
    Entropy H(X) = -∑ p(x) * log_base(p(x))
    
    Higher entropy indicates more uncertainty/randomness in the distribution.
    
    Args:
        probabilities: Probability distribution (must sum to 1)
        base: Logarithm base (2 for bits, e for nats, 10 for dits)
        epsilon: Small value to avoid log(0)
        
    Returns:
        Entropy value in the specified base
    """
    # Convert to numpy array if needed
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    probs = np.array(probabilities)
    
    # Normalize if not already (handle numerical errors)
    probs = probs / probs.sum()
    
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, epsilon, 1.0)
    
    # Compute entropy
    if base == 2.0:
        entropy = -np.sum(probs * np.log2(probs))
    elif base == np.e:
        entropy = -np.sum(probs * np.log(probs))
    else:
        entropy = -np.sum(probs * np.log(probs) / np.log(base))
    
    return float(entropy)


def compute_shannon_information(
    probability: float,
    base: float = 2.0,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the Shannon information (surprisal) of an observed event.
    
    Information I(x) = -log_base(p(x))
    
    This measures how "surprising" an event is. Rare events (low probability)
    have high information content.
    
    Args:
        probability: Probability of the observed event (0 < p <= 1)
        base: Logarithm base (2 for bits, e for nats, 10 for dits)
        epsilon: Small value to avoid log(0)
        
    Returns:
        Information content in the specified base
    """
    # Clip probability to avoid log(0)
    prob = max(epsilon, min(1.0, probability))
    
    # Compute information
    if base == 2.0:
        information = -np.log2(prob)
    elif base == np.e:
        information = -np.log(prob)
    else:
        information = -np.log(prob) / np.log(base)
    
    return float(information)


def compute_perplexity(
    probabilities: Union[List[float], np.ndarray],
    base: float = 2.0
) -> float:
    """
    Compute perplexity from a sequence of token probabilities.
    
    Perplexity = base^(average_surprisal) = base^(-1/N * ∑ log_base(p_i))
    
    Lower perplexity indicates the model is more confident in its predictions.
    
    Args:
        probabilities: List of probabilities for each token in sequence
        base: Logarithm base
        
    Returns:
        Perplexity value
    """
    probs = np.array(probabilities)
    log_probs = np.log(probs) / np.log(base)
    avg_log_prob = np.mean(log_probs)
    perplexity = base ** (-avg_log_prob)
    
    return float(perplexity)


def compute_cross_entropy(
    true_probs: Union[np.ndarray, torch.Tensor, List[float]],
    predicted_probs: Union[np.ndarray, torch.Tensor, List[float]],
    base: float = 2.0,
    epsilon: float = 1e-10
) -> float:
    """
    Compute cross-entropy between true and predicted distributions.
    
    H(P, Q) = -∑ p(x) * log_base(q(x))
    
    Args:
        true_probs: True probability distribution
        predicted_probs: Predicted probability distribution
        base: Logarithm base
        epsilon: Small value to avoid log(0)
        
    Returns:
        Cross-entropy value
    """
    # Convert to numpy arrays
    if isinstance(true_probs, torch.Tensor):
        true_probs = true_probs.cpu().numpy()
    if isinstance(predicted_probs, torch.Tensor):
        predicted_probs = predicted_probs.cpu().numpy()
    
    p = np.array(true_probs)
    q = np.array(predicted_probs)
    
    # Clip to avoid log(0)
    q = np.clip(q, epsilon, 1.0)
    
    # Compute cross-entropy
    if base == 2.0:
        ce = -np.sum(p * np.log2(q))
    elif base == np.e:
        ce = -np.sum(p * np.log(q))
    else:
        ce = -np.sum(p * np.log(q) / np.log(base))
    
    return float(ce)


def compute_kl_divergence(
    true_probs: Union[np.ndarray, torch.Tensor, List[float]],
    predicted_probs: Union[np.ndarray, torch.Tensor, List[float]],
    base: float = 2.0,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Kullback-Leibler divergence from predicted to true distribution.
    
    KL(P || Q) = ∑ p(x) * log_base(p(x) / q(x))
    
    Measures how different the predicted distribution is from the true distribution.
    
    Args:
        true_probs: True probability distribution (P)
        predicted_probs: Predicted probability distribution (Q)
        base: Logarithm base
        epsilon: Small value to avoid division by zero
        
    Returns:
        KL divergence value (always >= 0)
    """
    # Convert to numpy arrays
    if isinstance(true_probs, torch.Tensor):
        true_probs = true_probs.cpu().numpy()
    if isinstance(predicted_probs, torch.Tensor):
        predicted_probs = predicted_probs.cpu().numpy()
    
    p = np.array(true_probs)
    q = np.array(predicted_probs)
    
    # Clip to avoid division by zero
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Compute KL divergence
    if base == 2.0:
        kl = np.sum(p * np.log2(p / q))
    elif base == np.e:
        kl = np.sum(p * np.log(p / q))
    else:
        kl = np.sum(p * np.log(p / q) / np.log(base))
    
    return float(kl)


def analyze_token_information(
    token_probabilities: List[float],
    base: float = 2.0
) -> Dict:
    """
    Comprehensive information-theoretic analysis of a token sequence.
    
    Args:
        token_probabilities: List of probabilities for each generated token
        base: Logarithm base
        
    Returns:
        Dictionary containing:
            - 'surprisals': Information content for each token
            - 'mean_surprisal': Average surprisal across tokens
            - 'total_information': Sum of all information
            - 'perplexity': Perplexity of the sequence
            - 'min_surprisal': Minimum surprisal (most predictable token)
            - 'max_surprisal': Maximum surprisal (most surprising token)
    """
    surprisals = [compute_shannon_information(p, base) for p in token_probabilities]
    
    return {
        'surprisals': surprisals,
        'mean_surprisal': float(np.mean(surprisals)),
        'total_information': float(np.sum(surprisals)),
        'perplexity': compute_perplexity(token_probabilities, base),
        'min_surprisal': float(np.min(surprisals)),
        'max_surprisal': float(np.max(surprisals)),
        'std_surprisal': float(np.std(surprisals))
    }


def compute_varentropy(
    probabilities: Union[np.ndarray, torch.Tensor, List[float]],
    epsilon: float = 1e-10
) -> float:
    """
    Compute varentropy (variance of surprisal) for a probability distribution.
    
    Varentropy measures the variance in information content across the distribution.
    High varentropy indicates some outcomes are much more surprising than others.
    
    Args:
        probabilities: Probability distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        Varentropy value
    """
    # Convert to numpy array if needed
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    probs = np.array(probabilities)
    
    # Normalize
    probs = probs / probs.sum()
    probs = np.clip(probs, epsilon, 1.0)
    
    # Compute surprisals
    surprisals = -np.log2(probs)
    
    # Compute entropy (expected surprisal)
    entropy = np.sum(probs * surprisals)
    
    # Compute varentropy
    varentropy = np.sum(probs * (surprisals - entropy) ** 2)
    
    return float(varentropy)


def compute_mutual_information(
    joint_probs: np.ndarray,
    base: float = 2.0,
    epsilon: float = 1e-10
) -> float:
    """
    Compute mutual information I(X; Y) from joint probability distribution.
    
    I(X; Y) = ∑∑ p(x,y) * log_base(p(x,y) / (p(x) * p(y)))
    
    Args:
        joint_probs: 2D array of joint probabilities p(x, y)
        base: Logarithm base
        epsilon: Small value to avoid log(0)
        
    Returns:
        Mutual information value
    """
    joint_probs = np.clip(joint_probs, epsilon, 1.0)
    
    # Compute marginals
    px = joint_probs.sum(axis=1, keepdims=True)
    py = joint_probs.sum(axis=0, keepdims=True)
    
    # Compute mutual information
    independent_probs = px * py
    independent_probs = np.clip(independent_probs, epsilon, 1.0)
    
    if base == 2.0:
        mi = np.sum(joint_probs * np.log2(joint_probs / independent_probs))
    elif base == np.e:
        mi = np.sum(joint_probs * np.log(joint_probs / independent_probs))
    else:
        mi = np.sum(joint_probs * np.log(joint_probs / independent_probs) / np.log(base))
    
    return float(mi)
