# Task 04: Lattice-based WER Computation

## Overview
This system computes Word Error Rate (WER) using a lattice-based approach that handles cases where the reference transcription may contain errors. It constructs a lattice from multiple ASR model outputs and uses consensus to create a more reliable reference.

## Problem Statement

Given:
- Transcriptions from 5 ASR models for the same audio
- A human reference transcription that may contain errors

Goals:
- Construct a lattice capturing all valid transcription alternatives
- Handle insertions, deletions, and substitutions fairly
- Decide when to trust model agreement over the reference
- Compute WER that doesn't unfairly penalize correct models

## Approach

### 1. Lattice Construction

**Algorithm**:
```
1. Align all model outputs using dynamic programming
2. Find common anchor sequence (longest or reference)
3. Align each hypothesis to anchor
4. Group words by position
5. Create nodes for each unique word at each position
6. Create edges connecting consecutive positions
7. Calculate confidence based on model agreement
```

**Data Structure**:
- **Nodes**: (position, word, sources, confidence)
- **Edges**: (start_pos, end_pos, word, sources, weight)
- **Paths**: All possible routes through lattice

### 2. Handling Insertions, Deletions, Substitutions

**Insertions**:
- Model adds word not in reference
- If multiple models insert same word → likely correct
- Lattice includes insertion with high confidence
- Models with insertion get reduced penalty

**Deletions**:
- Model omits word from reference
- If multiple models omit → reference may be wrong
- Lattice consensus may exclude word
- Models without deletion get reduced penalty

**Substitutions**:
- Model uses different word than reference
- If multiple models agree on substitute → likely correct
- Lattice consensus uses majority word
- Models matching consensus get reduced penalty

**Fair Penalty Principle**:
- Don't penalize models for disagreeing with erroneous reference
- Use model agreement as signal for correctness
- Consensus transcription is more reliable than single reference

### 3. Trusting Models Over Reference

**Decision Criteria**:
```python
def should_trust_models(position, threshold=0.8):
    # Calculate model agreement at position
    max_confidence = max(confidence for all words at position)
    
    # Check if models agree strongly
    if max_confidence >= threshold:
        # Check if consensus differs from reference
        if consensus_word != reference_word:
            return True  # Trust models
    
    return False  # Trust reference
```

**Threshold**: 0.8 (80% of models must agree)

**Rationale**:
- High agreement (≥80%) suggests models are correct
- Low agreement suggests uncertainty, trust reference
- Balances between trusting models and reference

### 4. Alignment Unit Choice

**Chosen Unit: WORD**

**Justification**:

1. **Standard Metric**: WER = Word Error Rate (industry standard)
2. **Interpretability**: Words are meaningful units for humans
3. **Granularity**: Balances detail and robustness
4. **Conversational Speech**: Works well for natural dialogue
5. **Boundary Handling**: Natural word boundaries for insertions/deletions
6. **Compatibility**: Compatible with existing benchmarks and tools
7. **Language Agnostic**: Works across languages with word boundaries

**Alternatives Considered**:

- **Subword**: More granular, better for morphologically rich languages, but less interpretable
- **Phrase**: More semantic, but loses detail and harder to align
- **Character**: Too granular, sensitive to minor variations

**Conclusion**: Word-level provides the best balance for this task.

## Algorithm Details

### Multiple Sequence Alignment

```
Input: N model hypotheses + 1 reference
Output: Aligned sequences with positions

1. Choose anchor sequence (longest or reference)
2. For each other sequence:
   a. Initialize DP table (m+1 x n+1)
   b. Fill table with edit distances
   c. Backtrack to get alignment
   d. Record operations (match, substitute, insert, delete)
3. Merge alignments into common position space
```

### Consensus Generation

**Voting Strategy**:
```
For each position:
    Count votes for each word (1 vote per model)
    Select word with most votes
    Break ties by confidence or alphabetically
```

**Confidence Strategy**:
```
For each position:
    Calculate confidence = sources / total_models
    Select word with highest confidence
```

### WER Computation

**Standard WER**:
```
WER = (S + D + I) / N
where:
    S = substitutions
    D = deletions
    I = insertions
    N = reference length
```

**Lattice WER**:
```
1. Generate consensus transcription from lattice
2. Compute WER against consensus instead of reference
3. Compare with standard WER
4. Improvement = Standard WER - Lattice WER
```

## Implementation

### Core Classes

**WordLattice**:
- `build_from_hypotheses()`: Construct lattice from models
- `get_consensus_transcription()`: Generate consensus
- `should_trust_models_over_reference()`: Decision logic

**LatticeWER**:
- `compute_standard_wer()`: Traditional WER
- `compute_lattice_wer()`: Lattice-based WER
- `generate_report()`: Human-readable output

### Key Methods

```python
# Build lattice
lattice = WordLattice(alignment_unit='word')
lattice.build_from_hypotheses(hypotheses, reference)

# Get consensus
consensus = lattice.get_consensus_transcription(strategy='voting')

# Compute WER
wer_calc = LatticeWER(alignment_unit='word')
results = wer_calc.compute_lattice_wer(hypotheses, reference)
```

## Usage

### Installation

```bash
cd task_04
pip install -r requirements.txt
```

### Run Examples

```bash
cd src
python main.py
```

### Use in Code

```python
from lattice_wer import LatticeWER

# Your data
hypotheses = {
    'model_1': ['word1', 'word2', 'word3'],
    'model_2': ['word1', 'word2', 'word3'],
    # ... more models
}
reference = ['word1', 'word2', 'word3']

# Compute WER
wer_calc = LatticeWER(alignment_unit='word')
results = wer_calc.compute_lattice_wer(hypotheses, reference)

# Print report
print(wer_calc.generate_report(results))
```

## Examples

### Example 1: Reference Error

**Reference**: `['यह', 'बहुत', 'अच्छी', 'बात', 'है']` (Error: 'अच्छी' should be 'अच्छा')

**Models**:
- Model 1-4: `['यह', 'बहुत', 'अच्छा', 'बात', 'है']` (Correct)
- Model 5: `['यह', 'बहुत', 'अच्छी', 'बात', 'है']` (Matches wrong reference)

**Result**:
- Consensus: `['यह', 'बहुत', 'अच्छा', 'बात', 'है']` (4/5 vote)
- Models 1-4: WER improves (no longer penalized for being correct)
- Model 5: WER worsens (correctly penalized for matching error)

### Example 2: Insertion

**Reference**: `['मुझे', 'यह', 'पसंद', 'है']`

**Models**:
- Model 1, 3, 5: `['मुझे', 'यह', 'बहुत', 'पसंद', 'है']` (Insert 'बहुत')
- Model 2: `['मुझे', 'यह', 'पसंद', 'है']` (Matches reference)
- Model 4: `['मुझे', 'पसंद', 'है']` (Delete 'यह')

**Result**:
- Consensus: `['मुझे', 'यह', 'बहुत', 'पसंद', 'है']` (3/5 vote for 'बहुत')
- Models 1, 3, 5: WER improves (insertion was correct)
- Model 2: WER worsens (missed word)

## Output

### Console Report

```
================================================================================
LATTICE-BASED WER COMPUTATION REPORT
================================================================================

Alignment Unit: word
Reference Length: 5 words
Consensus Length: 5 words

--------------------------------------------------------------------------------
WER COMPARISON
--------------------------------------------------------------------------------
Model                Standard WER    Lattice WER     Improvement     Status    
--------------------------------------------------------------------------------
model_1_whisper      20.00%          0.00%           20.00%          ✓ Improved
model_2_wav2vec      20.00%          0.00%           20.00%          ✓ Improved
model_3_conformer    20.00%          0.00%           20.00%          ✓ Improved
model_4_deepspeech   20.00%          0.00%           20.00%          ✓ Improved
model_5_custom       0.00%           20.00%          -20.00%         Unchanged
```

### JSON Output

```json
{
  "model_1_whisper": {
    "standard_wer": 0.20,
    "lattice_wer": 0.00,
    "improvement": 0.20,
    "improved": true
  },
  "_meta": {
    "original_reference": ["यह", "बहुत", "अच्छी", "बात", "है"],
    "consensus_transcription": ["यह", "बहुत", "अच्छा", "बात", "है"],
    "alignment_unit": "word"
  }
}
```

## Theory

### Why Lattice-based WER?

**Problem with Standard WER**:
- Assumes reference is always correct
- Penalizes models for disagreeing with erroneous reference
- Unfair to models that are actually correct

**Lattice Solution**:
- Uses multiple models to validate reference
- Trusts consensus over single reference
- Fair evaluation even with imperfect reference

### Mathematical Foundation

**Consensus Probability**:
```
P(word | position) = count(models with word) / total_models
```

**Confidence Score**:
```
confidence = |sources| / |all_models|
```

**Trust Decision**:
```
trust_models = (confidence >= threshold) AND (consensus != reference)
```

### Complexity Analysis

**Time Complexity**:
- Alignment: O(N * M * L) where N=models, M=max_length, L=reference_length
- Lattice construction: O(N * M)
- Consensus: O(M * N)
- WER computation: O(M * L) per model
- Total: O(N * M * L)

**Space Complexity**:
- Lattice: O(M * N) for nodes and edges
- Alignments: O(N * M)
- Total: O(N * M)

## Limitations

1. **Requires Multiple Models**: Needs at least 3-5 models for reliable consensus
2. **Majority Bias**: Assumes majority is correct (may not always be true)
3. **No Semantic Understanding**: Purely lexical matching
4. **Word-level Only**: Current implementation doesn't support subword/phrase
5. **No Confidence Scores**: Doesn't use model confidence scores if available

## Future Improvements

1. **Weighted Voting**: Use model confidence scores in consensus
2. **Semantic Similarity**: Consider word embeddings for soft matching
3. **Language Model**: Use LM probability for consensus selection
4. **Subword Support**: Implement subword-level alignment
5. **Phrase Alignment**: Support phrase-level evaluation
6. **Acoustic Features**: Incorporate audio features for validation
7. **Active Learning**: Learn from corrections to improve consensus

## References

- Mangu, L., Brill, E., & Stolcke, A. (2000). Finding consensus in speech recognition: word error minimization and other applications of confusion networks.
- Fiscus, J. G. (1997). A post-processing system to yield reduced word error rates: Recognizer output voting error reduction (ROVER).
- Evermann, G., & Woodland, P. C. (2000). Posterior probability decoding, confidence estimation and system combination.

## Deliverables

✅ **Theory + Pseudocode/Code**: Complete implementation with detailed algorithms  
✅ **Lattice Construction**: Multi-sequence alignment with DP  
✅ **Fair Handling**: Insertions, deletions, substitutions handled fairly  
✅ **Trust Decision**: Threshold-based model agreement vs reference  
✅ **Alignment Unit**: Word-level with justification  
✅ **WER Computation**: Both standard and lattice-based WER  
✅ **Examples**: 3 examples demonstrating different scenarios  
✅ **Documentation**: Comprehensive theory and implementation docs  
