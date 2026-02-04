# Why Entropy Changes (or Doesn't) Across a Sequence

## Your Question: Why doesn't entropy change?

In the **MOCK version**, entropy was constant at 3.6147 bits because the probability 
distribution had the same shape at every position.

In **REAL language models**, entropy varies significantly based on context!

## Comparison: Mock vs Real

### Mock Version (entropy_intervention_example.py)
```
Position   Token         Entropy (bits)  Information (bits)  
---------------------------------------------------------------
1          hello         3.6147          3.2532              
2          what          3.6147          2.8439              
3          is            3.6147          3.8777              
4          the           3.6147          3.3020              
5          entropy       3.6147          4.7953              
```
**Problem**: Same distribution shape → Same entropy every time

### Real Version (entropy_real_probabilities.py with GPT-2)
```
Position   Token         Entropy (bits)  Information (bits)  
---------------------------------------------------------------
1          hello         10.1250         15.5202              
2           what          8.8984         10.7077              
3           is            6.7695          4.7123              
4           the           6.9609          3.2401              
5           entropy      10.4609         17.2858              
6           and           5.7539          6.5172              
7           information   7.8867         10.2426              
8           of            7.1797          3.3013              
9           this          5.8242          3.5293              
10          string       10.1328          8.2594              
```
**Real behavior**: Entropy varies from 5.75 to 10.46 bits!

## Why Does Real Entropy Vary?

### High Entropy Positions (≈10 bits)
- **Position 1** ("hello"): 10.13 bits
  - At the start, many possible first words
  - Model is very uncertain
  
- **Position 5** (" entropy"): 10.46 bits
  - After "hello what is the...", could be many things
  - Technical word "entropy" is uncommon (surprising!)

### Low Entropy Positions (≈6 bits)
- **Position 3** (" is"): 6.77 bits
  - After "hello what", next word is more predictable
  - Common pattern: "what" often followed by "is", "are", "do"
  
- **Position 6** (" and"): 5.75 bits (LOWEST)
  - After "entropy", likely continuation words
  - "and", "is", "of" are predictable

## Information (Surprisal) Tells the Story

**High Information = Surprising Token**
- Position 5 (" entropy"): 17.29 bits
  - Very rare word in this context
  - Model gave it only 0.0006% probability!
  
- Position 1 ("hello"): 15.52 bits
  - Uncommon way to start a sentence
  - Model expected "\n", "The", or "A"

**Low Information = Expected Token**
- Position 4 (" the"): 3.24 bits
  - Very predictable after "what is"
  - Model gave it 10.6% probability

- Position 9 (" this"): 3.53 bits
  - Natural phrase: "of this"
  - Model confidence: 8.7%

## Key Insights

1. **Entropy measures uncertainty** before seeing the token
   - High entropy = many possible next tokens
   - Low entropy = model is confident about what comes next

2. **Information measures surprise** after seeing the token
   - High information = unlikely/surprising token chosen
   - Low information = expected/predictable token chosen

3. **Context matters**
   - "hello what is THE..." → low entropy (predictable)
   - "...is the ENTROPY..." → high information (surprising word)

## Total Statistics

**Mock Version:**
- Average entropy: 3.61 bits (constant)
- Total information: 37.84 bits

**Real Version (GPT-2):**
- Average entropy: 8.00 bits (varies 5.75-10.46)
- Total information: 83.32 bits
- Perplexity: 322.15

The real model shows **2x more information** because it has actual 
probability distributions that vary based on context!

## To Get Real Results

Use the transformers version:
```bash
source .venv/bin/activate
python examples/entropy_real_probabilities.py
```

Or adapt it to use your local Llama model through transformers.
