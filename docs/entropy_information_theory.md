# Information Theory for Language Models: Computing Entropy and Surprisal

## Overview

This document explains the mathematical foundations for computing entropy and information in language model generation, using the example: **"hello what is the entropy and information of this string"**

## 1. Core Concepts

### 1.1 Shannon Entropy

**Entropy** measures the **uncertainty** or **randomness** in a probability distribution before observing an outcome.

**Formula:**
$$H(P) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

Where:
- $P(x_i)$ = probability of token $x_i$
- $n$ = vocabulary size
- $\log_2$ = logarithm base 2 (measures in **bits**)

**Interpretation:**
- **High entropy** (e.g., 10 bits): Many equally likely outcomes → high uncertainty
- **Low entropy** (e.g., 1 bit): Few likely outcomes → low uncertainty
- **Zero entropy**: Only one outcome is possible (deterministic)

### 1.2 Shannon Information (Surprisal)

**Information** measures how **surprising** a specific observed token is, given the probability distribution.

**Formula:**
$$I(x) = -\log_2 P(x)$$

Where:
- $P(x)$ = probability the model assigned to the observed token
- Result is in **bits**

**Interpretation:**
- **High information** (e.g., 15 bits): Token had very low probability → very surprising
- **Low information** (e.g., 2 bits): Token had high probability → expected
- **Zero information**: Token had probability = 1 (certain)

**Key Relationship:**
- If $P(x) = 0.5$ → $I(x) = 1$ bit (like a fair coin flip)
- If $P(x) = 0.25$ → $I(x) = 2$ bits (like choosing 1 of 4 equally likely options)
- If $P(x) = 1.0$ → $I(x) = 0$ bits (no surprise, already known)

## 2. Token-by-Token Analysis Process

For a target string $s = [t_1, t_2, ..., t_n]$, we analyze generation step-by-step:

### Step $i$: Analyzing Token $t_i$

**Given:**
- Prompt: $p$ (can be empty)
- Context so far: $c_{i-1} = [t_1, t_2, ..., t_{i-1}]$

**Process:**

1. **Query the model** with $p + c_{i-1}$ to get probability distribution:
   $$P_i(·) = \text{Model}(p + c_{i-1})$$
   
   This gives probabilities for all possible next tokens:
   $$P_i = [P_i(x_1), P_i(x_2), ..., P_i(x_V)]$$
   where $V$ is vocabulary size

2. **Compute entropy** of the distribution $P_i$:
   $$H_i = -\sum_{j=1}^{V} P_i(x_j) \log_2 P_i(x_j)$$
   
   This measures: *"How uncertain is the model about what token comes next?"*

3. **Extract probability** of the actual token $t_i$ that appears in our string:
   $$p_i = P_i(t_i)$$

4. **Compute information** (surprisal) of that token:
   $$I_i = -\log_2(p_i)$$
   
   This measures: *"How surprising was the actual token $t_i$?"*

5. **Update context**: $c_i = c_{i-1} + t_i$

6. **Repeat** for next token

## 3. Worked Example

### Example String: "hello what is the entropy and information of this string"

Let's trace through the first few tokens:

---

#### Token 1: "hello"

**Context:** Empty (starting from scratch)
**Model input:** "" (or BOS token)

**Model outputs probability distribution $P_1$:**
```
P₁("The")   = 0.037
P₁("I")     = 0.018
P₁("A")     = 0.019
P₁("hello") = 0.000021  ← our token
...
(50,000 more tokens)
```

**Compute Entropy $H_1$:**
$$H_1 = -\sum_{j=1}^{50000} P_1(x_j) \log_2 P_1(x_j) \approx 10.13 \text{ bits}$$

**Interpretation:** High entropy means the model is very uncertain about what the first word should be (many possibilities).

**Extract probability of "hello":**
$$p_1 = P_1(\text{"hello"}) = 0.000021$$

**Compute Information $I_1$:**
$$I_1 = -\log_2(0.000021) = 15.52 \text{ bits}$$

**Interpretation:** "hello" is very surprising as a starting word (only 0.002% probability!).

---

#### Token 2: " what"

**Context:** "hello"
**Model input:** "hello"

**Model outputs $P_2$:**
```
P₂(",")     = 0.129
P₂(".")     = 0.094
P₂(" there")= 0.025
P₂(" what") = 0.000598  ← our token
...
```

**Compute Entropy $H_2$:**
$$H_2 = -\sum_{j} P_2(x_j) \log_2 P_2(x_j) \approx 8.90 \text{ bits}$$

**Interpretation:** Still fairly uncertain, but less than before. Model expects punctuation or common greeting words.

**Extract probability:**
$$p_2 = P_2(\text{" what"}) = 0.000598$$

**Compute Information $I_2$:**
$$I_2 = -\log_2(0.000598) = 10.71 \text{ bits}$$

**Interpretation:** " what" is also surprising here. Model expected "hello" to be followed by punctuation or "there".

---

#### Token 3: " is"

**Context:** "hello what"
**Model input:** "hello what"

**Model outputs $P_3$:**
```
P₃("'s")    = 0.086
P₃(" you")  = 0.081
P₃(" I")    = 0.081
P₃(" is")   = 0.038  ← our token (much higher!)
...
```

**Compute Entropy $H_3$:**
$$H_3 = -\sum_{j} P_3(x_j) \log_2 P_3(x_j) \approx 6.77 \text{ bits}$$

**Interpretation:** Lower entropy! After "hello what", the model has stronger expectations.

**Extract probability:**
$$p_3 = P_3(\text{" is"}) = 0.038$$

**Compute Information $I_3$:**
$$I_3 = -\log_2(0.038) = 4.71 \text{ bits}$$

**Interpretation:** Much less surprising! " is" is a reasonable continuation of "what".

---

#### Token 4: " the"

**Context:** "hello what is"
**Model input:** "hello what is"

**Model outputs $P_4$:**
```
P₄(" this") = 0.136
P₄(" it")   = 0.113
P₄(" the")  = 0.106  ← our token (top 3!)
P₄(" that") = 0.064
...
```

**Compute Entropy $H_4$:**
$$H_4 \approx 6.96 \text{ bits}$$

**Interpretation:** Similar to previous step—model knows we're in a question structure.

**Extract probability:**
$$p_4 = P_4(\text{" the"}) = 0.106$$

**Compute Information $I_4$:**
$$I_4 = -\log_2(0.106) = 3.24 \text{ bits}$$

**Interpretation:** Very predictable! " the" is a top-3 choice (10.6% probability).

---

#### Token 5: " entropy"

**Context:** "hello what is the"
**Model input:** "hello what is the"

**Model outputs $P_5$:**
```
P₅(" difference") = 0.051
P₅(" best")       = 0.033
P₅(" problem")    = 0.031
P₅(" entropy")    = 0.000006  ← our token (very rare!)
...
```

**Compute Entropy $H_5$:**
$$H_5 \approx 10.46 \text{ bits}$$

**Interpretation:** High entropy! After "what is the", many nouns are possible.

**Extract probability:**
$$p_5 = P_5(\text{" entropy"}) = 0.000006$$

**Compute Information $I_5$:**
$$I_5 = -\log_2(0.000006) = 17.29 \text{ bits}$$

**Interpretation:** VERY surprising! "entropy" is a technical term the model didn't expect (0.0006% probability).

This is the **highest information token** in our sequence!

---

## 4. Aggregate Metrics

After analyzing all $n$ tokens, we compute summary statistics:

### 4.1 Average Entropy

$$\bar{H} = \frac{1}{n} \sum_{i=1}^{n} H_i$$

**For our example:**
$$\bar{H} = \frac{10.13 + 8.90 + 6.77 + 6.96 + 10.46 + 5.75 + 7.89 + 7.18 + 5.82 + 10.13}{10} = 8.00 \text{ bits}$$

**Interpretation:** On average, the model had about 8 bits of uncertainty at each position (equivalent to choosing from $2^8 = 256$ equally likely options).

### 4.2 Total Entropy of the Sequence

$$H_{\text{total}} = \sum_{i=1}^{n} H_i$$

**For our example:**
$$H_{\text{total}} = 10.13 + 8.90 + ... + 10.13 = 79.99 \text{ bits}$$

**Interpretation:** This IS the entropy of the entire sequence under the autoregressive model!

**Important: This is the true joint entropy!**

Using the **chain rule of entropy**, the entropy of a sequence can be decomposed as:

$$H(T_1, T_2, ..., T_n) = H(T_1) + H(T_2|T_1) + H(T_3|T_1,T_2) + ... + H(T_n|T_{1:n-1})$$

where:
- $H(T_1)$ = entropy of first token (no context)
- $H(T_i|T_{1:i-1})$ = conditional entropy of token $i$ given all previous tokens

Each $H_i$ we compute is exactly $H(T_i|T_{1:i-1})$ - the conditional entropy at position $i$.

Therefore:
$$H_{\text{sequence}} = \sum_{i=1}^{n} H_i = 79.99 \text{ bits}$$

**What this means:**
- If we randomly sample a 10-token sequence from this model's distribution, we expect about 80 bits of uncertainty on average
- This represents the "information capacity" of sequences of this length from this model
- Higher entropy = more diverse/unpredictable sequences possible
- Lower entropy = more stereotyped/predictable sequences

**Contrast with Information:**
- **Entropy** (79.99 bits): How uncertain we are about which sequence we'll see
- **Information** (83.32 bits): How surprising the specific sequence we observed was

The observed sequence has slightly more information than the average entropy because it contained some unlikely tokens (like "entropy").

### 4.3 Total Information (Sequence Surprisal)

$$I_{\text{total}} = \sum_{i=1}^{n} I_i = -\sum_{i=1}^{n} \log_2 P(t_i | t_{1:i-1})$$

**For our example:**
$$I_{\text{total}} = 15.52 + 10.71 + 4.71 + 3.24 + 17.29 + 6.52 + 10.24 + 3.30 + 3.53 + 8.26 = 83.32 \text{ bits}$$

**Interpretation:** This is the **total surprisal** of the sequence. It represents:
- How many bits would be needed to encode this sequence given the model's predictions
- How "unlikely" the sequence was according to the model
- Lower values = more typical/expected sequence
- Higher values = more unusual/surprising sequence

**Important relationship:**
$$I_{\text{total}} = -\log_2 P(t_1, t_2, ..., t_n)$$

where $P(t_1, t_2, ..., t_n)$ is the probability the model assigns to the entire sequence.

Therefore:
$$P(\text{sequence}) = 2^{-I_{\text{total}}} = 2^{-83.32} \approx 1.03 \times 10^{-25}$$

The sequence has an extremely low probability!

### 4.4 Average Information per Token

$$\bar{I} = \frac{1}{n} \sum_{i=1}^{n} I_i = \frac{I_{\text{total}}}{n}$$

**For our example:**
$$\bar{I} = \frac{83.32}{10} = 8.33 \text{ bits/token}$$

**Interpretation:** On average, each token carried 8.33 bits of information (surprisal).

### 4.5 Perplexity

Perplexity is an alternative way to express average surprisal:

$$\text{Perplexity} = 2^{\bar{I}}$$

**For our example:**
$$\text{Perplexity} = 2^{8.33} = 322.15$$

**Interpretation:** The model is as uncertain as if it had to choose uniformly from 322 options at each step.

- **Lower perplexity** (e.g., 10) = model is confident, few likely options
- **Higher perplexity** (e.g., 1000) = model is uncertain, many options seem possible

## 5. Key Relationships

### 5.1 Entropy vs. Information

At each position $i$:

- **Entropy $H_i$**: Uncertainty *before* seeing the token
  - Property of the entire distribution $P_i$
  - Measures the model's uncertainty
  
- **Information $I_i$**: Surprise *after* seeing the token
  - Property of the specific observed token $t_i$
  - Measures how unexpected that token was

**Expected Information equals Entropy:**
$$\mathbb{E}_{t \sim P_i}[I_i(t)] = H_i$$

If we average the surprisal of all possible tokens (weighted by their probability), we get the entropy!

### 5.2 Why Information Can Be Higher or Lower Than Entropy

**Information can be lower than entropy:**
- If we observe a high-probability token: $I_i < H_i$
- Example: At position 4, $H_4 = 6.96$ bits but $I_4 = 3.24$ bits
- We got lucky and observed a likely token!

**Information can be higher than entropy:**
- If we observe a low-probability token: $I_i > H_i$
- Example: At position 5, $H_5 = 10.46$ bits but $I_5 = 17.29$ bits
- We observed a very unlikely token ("entropy")

**Over many observations:**
- Average information converges to entropy
- But for a single sequence, they can differ significantly

### 5.3 Minimum Encoding Length

The total information $I_{\text{total}}$ represents the **minimum number of bits** needed to encode the sequence using an optimal code based on the model's predictions.

This is the foundation of:
- **Arithmetic coding**
- **Language model compression**
- **Cross-entropy loss** in neural network training

## 6. Practical Applications

### 6.1 Model Evaluation

- **Lower perplexity** = better model (more accurate predictions)
- Compare perplexity across different models on the same text

### 6.2 Text Analysis

- **High-information tokens** reveal surprising/creative word choices
- **Low-information tokens** show predictable/formulaic patterns
- Can identify unusual vs. typical text

### 6.3 Generation Quality

- **High entropy** at a position = model is uncertain → generation may be incoherent
- **Low entropy** = model is confident → generation likely coherent
- Can use entropy to detect when model is "confused"

### 6.4 Intervention Strategies

- **High entropy positions**: Good places to intervene (many options)
- **Low entropy positions**: Hard to intervene (strong expectations)
- Can target high-information tokens to make text more surprising/creative

## 7. Mathematical Properties

### 7.1 Bounds on Entropy

$$0 \leq H(P) \leq \log_2(V)$$

where $V$ is vocabulary size.

- **Minimum (0 bits):** All probability on one token (deterministic)
- **Maximum ($\log_2(V)$ bits):** Uniform distribution over all $V$ tokens

For $V = 50,000$:
$$H_{\max} = \log_2(50000) \approx 15.61 \text{ bits}$$

### 7.2 Bounds on Information

$$0 \leq I(x) \leq \infty$$

- **Minimum (0 bits):** Token has probability 1 (certain)
- **No upper bound:** As $P(x) \to 0$, $I(x) \to \infty$

Practically, information is bounded by how small the model's probabilities can be.

### 7.3 Chain Rule for Entropy and Information

**Entropy Chain Rule:**
$$H(T_1, T_2, ..., T_n) = \sum_{i=1}^{n} H(T_i | T_{1:i-1})$$

This decomposes the joint entropy into conditional entropies at each position.

**For autoregressive language models**, this is exactly what we compute:
$$H_{\text{sequence}} = H_1 + H_2 + ... + H_n$$

where each $H_i$ is the entropy of the model's distribution at position $i$.

**Information Chain Rule:**
$$I_{\text{total}} = I(t_1) + I(t_2|t_1) + I(t_3|t_1,t_2) + ... + I(t_n|t_{1:n-1})$$

Each term is the surprisal of a token given all previous tokens.

This equals:
$$I_{\text{total}} = -\log_2 P(t_1, t_2, ..., t_n)$$

**Key Insight:**
- The sum of conditional entropies = entropy of the sequence
- The sum of conditional information = negative log probability of the sequence
- Both decompose naturally for autoregressive models!

**Relationship:**
$$\mathbb{E}[\text{Information}] = \text{Entropy}$$

Over all possible sequences:
$$\mathbb{E}_{s \sim P}[I_{\text{total}}(s)] = H_{\text{sequence}}$$

For our specific sequence:
- Expected information: $H_{\text{sequence}} = 79.99$ bits
- Observed information: $I_{\text{total}} = 83.32$ bits
- Our sequence was 3.33 bits more surprising than average!

## 8. Comparison: Mock vs. Real Probabilities

### Mock Version (Constant Entropy)
```
Position   Token         Entropy    Information  
1          hello         3.61 bits  3.25 bits
2          what          3.61 bits  2.84 bits
3          is            3.61 bits  3.88 bits
...
Average:                 3.61 bits  3.78 bits
```

**Issue:** Same distribution shape at every position (unrealistic)

### Real Version (GPT-2)
```
Position   Token         Entropy    Information  
1          hello        10.13 bits  15.52 bits
2           what         8.90 bits  10.71 bits
3           is           6.77 bits   4.71 bits
4           the          6.96 bits   3.24 bits
5           entropy     10.46 bits  17.29 bits
...
Average:                 8.00 bits   8.33 bits
```

**Real behavior:** Entropy varies based on context!

## 9. Summary

### Core Formulas

**Entropy (uncertainty before observation):**
$$H = -\sum_i P(x_i) \log_2 P(x_i)$$

**Information (surprisal after observation):**
$$I(x) = -\log_2 P(x)$$

**Total Information (sequence):**
$$I_{\text{total}} = \sum_{i=1}^{n} I_i = -\log_2 P(\text{sequence})$$

**Perplexity:**
$$\text{PPL} = 2^{\bar{I}}$$

### Key Insights

1. **Entropy** measures the model's uncertainty at each step
2. **Information** measures how surprising each observed token is
3. **Total information** quantifies the overall likelihood of a sequence
4. **Context matters**: Real models have varying entropy based on what came before
5. **Rare/technical words** (like "entropy") have high information (surprise)
6. **Common/predictable words** (like "the") have low information

### Implementation

See the example scripts:
- `examples/entropy_intervention_example.py` - Framework with mock probabilities
- `examples/entropy_real_probabilities.py` - Real probabilities from GPT-2/Llama

The theoretical foundations here apply to any probabilistic language model!
