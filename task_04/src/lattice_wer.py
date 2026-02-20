"""
Lattice-based WER Computation System
Handles multiple ASR model outputs with potentially erroneous reference transcriptions
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import json
from dataclasses import dataclass
import editdistance

@dataclass
class LatticeNode:
    """Node in the word lattice"""
    position: int
    word: str
    sources: Set[str]  # Which models contributed this word
    confidence: float
    
@dataclass
class LatticeEdge:
    """Edge connecting lattice nodes"""
    start_pos: int
    end_pos: int
    word: str
    sources: Set[str]
    weight: float

class WordLattice:
    """
    Word lattice structure capturing all valid transcription alternatives
    from multiple ASR model outputs
    """
    
    def __init__(self, alignment_unit: str = 'word'):
        """
        Initialize lattice
        
        Args:
            alignment_unit: 'word', 'subword', or 'phrase'
        """
        self.alignment_unit = alignment_unit
        self.nodes = []  # List of LatticeNode
        self.edges = []  # List of LatticeEdge
        self.paths = []  # All possible paths through lattice
        
    def build_from_hypotheses(self, hypotheses: Dict[str, List[str]], 
                              reference: List[str]) -> None:
        """
        Build lattice from multiple ASR hypotheses and reference
        
        Args:
            hypotheses: Dict mapping model_name -> list of words
            reference: Reference transcription (may contain errors)
        """
        # Align all hypotheses using dynamic programming
        alignments = self._align_all_sequences(hypotheses, reference)
        
        # Build lattice structure from alignments
        self._construct_lattice(alignments, reference)
        
    def _align_all_sequences(self, hypotheses: Dict[str, List[str]], 
                            reference: List[str]) -> Dict:
        """
        Align all hypotheses to each other and to reference
        Uses multiple sequence alignment approach
        """
        all_sequences = {**hypotheses, 'reference': reference}
        
        # Find longest sequence as anchor
        anchor_name = max(all_sequences.keys(), key=lambda k: len(all_sequences[k]))
        anchor_seq = all_sequences[anchor_name]
        
        # Align each sequence to anchor
        alignments = {}
        for name, seq in all_sequences.items():
            if name == anchor_name:
                alignments[name] = [(i, word, 'match') for i, word in enumerate(seq)]
            else:
                alignments[name] = self._align_pair(anchor_seq, seq, name)
        
        return alignments
    
    def _align_pair(self, seq1: List[str], seq2: List[str], 
                   name: str) -> List[Tuple]:
        """
        Align two sequences using dynamic programming
        Returns alignment with positions and operations
        """
        m, n = len(seq1), len(seq2)
        
        # DP table: dp[i][j] = (cost, operation)
        dp = [[None for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = (i, 'delete')
        for j in range(n + 1):
            dp[0][j] = (j, 'insert')
        dp[0][0] = (0, 'start')
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/substitute
                if seq1[i-1].lower() == seq2[j-1].lower():
                    match_cost = dp[i-1][j-1][0]
                    dp[i][j] = (match_cost, 'match')
                else:
                    sub_cost = dp[i-1][j-1][0] + 1
                    del_cost = dp[i-1][j][0] + 1
                    ins_cost = dp[i][j-1][0] + 1
                    
                    min_cost = min(sub_cost, del_cost, ins_cost)
                    if min_cost == sub_cost:
                        dp[i][j] = (sub_cost, 'substitute')
                    elif min_cost == del_cost:
                        dp[i][j] = (del_cost, 'delete')
                    else:
                        dp[i][j] = (ins_cost, 'insert')
        
        # Backtrack to get alignment
        alignment = []
        i, j = m, n
        pos = max(m, n)
        
        while i > 0 or j > 0:
            if i == 0:
                alignment.append((j-1, seq2[j-1], 'insert'))
                j -= 1
            elif j == 0:
                alignment.append((i-1, seq1[i-1], 'delete'))
                i -= 1
            else:
                op = dp[i][j][1]
                if op == 'match':
                    alignment.append((i-1, seq2[j-1], 'match'))
                    i -= 1
                    j -= 1
                elif op == 'substitute':
                    alignment.append((i-1, seq2[j-1], 'substitute'))
                    i -= 1
                    j -= 1
                elif op == 'delete':
                    alignment.append((i-1, seq1[i-1], 'delete'))
                    i -= 1
                elif op == 'insert':
                    alignment.append((j-1, seq2[j-1], 'insert'))
                    j -= 1
        
        alignment.reverse()
        return alignment
    
    def _construct_lattice(self, alignments: Dict, reference: List[str]) -> None:
        """
        Construct lattice structure from alignments
        """
        # Group words by position
        position_words = defaultdict(lambda: defaultdict(set))
        
        for model_name, alignment in alignments.items():
            for pos, word, op in alignment:
                if op != 'delete':  # Skip deletions
                    position_words[pos][word.lower()].add(model_name)
        
        # Create nodes and edges
        positions = sorted(position_words.keys())
        
        for i, pos in enumerate(positions):
            words_at_pos = position_words[pos]
            
            for word, sources in words_at_pos.items():
                # Calculate confidence based on model agreement
                confidence = len(sources) / (len(alignments))
                
                node = LatticeNode(
                    position=pos,
                    word=word,
                    sources=sources,
                    confidence=confidence
                )
                self.nodes.append(node)
                
                # Create edges to next position
                if i < len(positions) - 1:
                    next_pos = positions[i + 1]
                    edge = LatticeEdge(
                        start_pos=pos,
                        end_pos=next_pos,
                        word=word,
                        sources=sources,
                        weight=1.0 - confidence  # Lower weight = higher confidence
                    )
                    self.edges.append(edge)
    
    def get_consensus_transcription(self, strategy: str = 'voting') -> List[str]:
        """
        Get consensus transcription from lattice
        
        Args:
            strategy: 'voting' (majority vote) or 'confidence' (highest confidence path)
        
        Returns:
            List of words forming consensus transcription
        """
        if strategy == 'voting':
            return self._voting_consensus()
        elif strategy == 'confidence':
            return self._confidence_consensus()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _voting_consensus(self) -> List[str]:
        """Get consensus using majority voting at each position"""
        position_words = defaultdict(list)
        
        for node in self.nodes:
            position_words[node.position].append((node.word, len(node.sources)))
        
        consensus = []
        for pos in sorted(position_words.keys()):
            words = position_words[pos]
            # Pick word with most sources (votes)
            best_word = max(words, key=lambda x: x[1])[0]
            consensus.append(best_word)
        
        return consensus
    
    def _confidence_consensus(self) -> List[str]:
        """Get consensus using highest confidence path"""
        position_words = defaultdict(list)
        
        for node in self.nodes:
            position_words[node.position].append((node.word, node.confidence))
        
        consensus = []
        for pos in sorted(position_words.keys()):
            words = position_words[pos]
            # Pick word with highest confidence
            best_word = max(words, key=lambda x: x[1])[0]
            consensus.append(best_word)
        
        return consensus
    
    def should_trust_models_over_reference(self, position: int, 
                                          threshold: float = 0.8) -> bool:
        """
        Decide whether to trust model agreement over reference at a position
        
        Args:
            position: Position in lattice
            threshold: Agreement threshold (0.0-1.0)
        
        Returns:
            True if models agree strongly and differ from reference
        """
        nodes_at_pos = [n for n in self.nodes if n.position == position]
        
        if not nodes_at_pos:
            return False
        
        # Check if models agree (high confidence)
        max_confidence = max(n.confidence for n in nodes_at_pos)
        
        # Check if reference differs from consensus
        reference_word = None
        consensus_word = None
        
        for node in nodes_at_pos:
            if 'reference' in node.sources:
                reference_word = node.word
            if node.confidence == max_confidence:
                consensus_word = node.word
        
        # Trust models if they agree strongly and differ from reference
        if max_confidence >= threshold and reference_word != consensus_word:
            return True
        
        return False


class LatticeWER:
    """
    Compute WER using lattice-based transcription
    Handles cases where reference may be incorrect
    """
    
    def __init__(self, alignment_unit: str = 'word'):
        """
        Initialize WER calculator
        
        Args:
            alignment_unit: 'word', 'subword', or 'phrase'
        """
        self.alignment_unit = alignment_unit
        self.alignment_justification = self._justify_alignment_unit()
    
    def _justify_alignment_unit(self) -> str:
        """Provide justification for chosen alignment unit"""
        if self.alignment_unit == 'word':
            return """
            Word-level alignment chosen because:
            1. Standard metric for ASR evaluation (WER = Word Error Rate)
            2. Interpretable and meaningful for human evaluation
            3. Balances granularity and robustness
            4. Works well for conversational speech
            5. Handles insertions/deletions naturally at word boundaries
            6. Compatible with existing benchmarks and tools
            """
        elif self.alignment_unit == 'subword':
            return """
            Subword-level alignment chosen because:
            1. More granular than words, captures partial matches
            2. Better for morphologically rich languages
            3. Handles OOV words more gracefully
            4. Aligns with modern tokenization (BPE, WordPiece)
            """
        elif self.alignment_unit == 'phrase':
            return """
            Phrase-level alignment chosen because:
            1. Captures semantic units beyond single words
            2. More robust to word order variations
            3. Better for meaning-preserving evaluation
            4. Reduces impact of minor word-level differences
            """
        else:
            return "Custom alignment unit"
    
    def compute_standard_wer(self, hypothesis: List[str], 
                            reference: List[str]) -> Dict:
        """
        Compute standard WER between hypothesis and reference
        
        Returns:
            Dict with WER, substitutions, deletions, insertions
        """
        # Use edit distance
        distance = editdistance.eval(hypothesis, reference)
        
        # Get detailed alignment
        alignment = self._get_alignment_details(hypothesis, reference)
        
        wer = distance / len(reference) if len(reference) > 0 else 0.0
        
        return {
            'wer': wer,
            'distance': distance,
            'reference_length': len(reference),
            'substitutions': alignment['substitutions'],
            'deletions': alignment['deletions'],
            'insertions': alignment['insertions']
        }
    
    def _get_alignment_details(self, hyp: List[str], 
                              ref: List[str]) -> Dict:
        """Get detailed alignment with operation counts"""
        m, n = len(hyp), len(ref)
        
        # DP for edit distance with backtracking
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        ops = [['' for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
            ops[i][0] = 'D'
        for j in range(n + 1):
            dp[0][j] = j
            ops[0][j] = 'I'
        ops[0][0] = ''
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp[i-1].lower() == ref[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = 'M'
                else:
                    sub = dp[i-1][j-1] + 1
                    delete = dp[i-1][j] + 1
                    insert = dp[i][j-1] + 1
                    
                    min_cost = min(sub, delete, insert)
                    dp[i][j] = min_cost
                    
                    if min_cost == sub:
                        ops[i][j] = 'S'
                    elif min_cost == delete:
                        ops[i][j] = 'D'
                    else:
                        ops[i][j] = 'I'
        
        # Count operations
        i, j = m, n
        subs, dels, ins = 0, 0, 0
        
        while i > 0 or j > 0:
            if i == 0:
                ins += 1
                j -= 1
            elif j == 0:
                dels += 1
                i -= 1
            else:
                op = ops[i][j]
                if op == 'M':
                    i -= 1
                    j -= 1
                elif op == 'S':
                    subs += 1
                    i -= 1
                    j -= 1
                elif op == 'D':
                    dels += 1
                    i -= 1
                else:  # I
                    ins += 1
                    j -= 1
        
        return {
            'substitutions': subs,
            'deletions': dels,
            'insertions': ins
        }
    
    def compute_lattice_wer(self, hypotheses: Dict[str, List[str]], 
                           reference: List[str],
                           trust_threshold: float = 0.8) -> Dict:
        """
        Compute WER using lattice-based approach
        
        Args:
            hypotheses: Dict mapping model_name -> transcription
            reference: Reference transcription (may have errors)
            trust_threshold: Threshold for trusting models over reference
        
        Returns:
            Dict with WER for each model using lattice-based reference
        """
        # Build lattice
        lattice = WordLattice(alignment_unit=self.alignment_unit)
        lattice.build_from_hypotheses(hypotheses, reference)
        
        # Get consensus transcription
        consensus = lattice.get_consensus_transcription(strategy='voting')
        
        # Compute WER for each model against both reference and consensus
        results = {}
        
        for model_name, hypothesis in hypotheses.items():
            # Standard WER (against original reference)
            standard_wer = self.compute_standard_wer(hypothesis, reference)
            
            # Lattice WER (against consensus)
            lattice_wer = self.compute_standard_wer(hypothesis, consensus)
            
            # Determine if lattice improved WER
            improvement = standard_wer['wer'] - lattice_wer['wer']
            
            results[model_name] = {
                'standard_wer': standard_wer,
                'lattice_wer': lattice_wer,
                'improvement': improvement,
                'improved': improvement > 0,
                'hypothesis_length': len(hypothesis)
            }
        
        # Add consensus and lattice info
        results['_meta'] = {
            'original_reference': reference,
            'consensus_transcription': consensus,
            'reference_length': len(reference),
            'consensus_length': len(consensus),
            'alignment_unit': self.alignment_unit,
            'trust_threshold': trust_threshold
        }
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable report"""
        report = []
        report.append("=" * 80)
        report.append("LATTICE-BASED WER COMPUTATION REPORT")
        report.append("=" * 80)
        
        meta = results.get('_meta', {})
        
        report.append(f"\nAlignment Unit: {meta.get('alignment_unit', 'word')}")
        report.append(f"Reference Length: {meta.get('reference_length', 0)} words")
        report.append(f"Consensus Length: {meta.get('consensus_length', 0)} words")
        
        report.append("\n" + "-" * 80)
        report.append("WER COMPARISON")
        report.append("-" * 80)
        
        # Table header
        report.append(f"{'Model':<20} {'Standard WER':<15} {'Lattice WER':<15} {'Improvement':<15} {'Status':<10}")
        report.append("-" * 80)
        
        for model_name, model_results in results.items():
            if model_name == '_meta':
                continue
            
            std_wer = model_results['standard_wer']['wer']
            lat_wer = model_results['lattice_wer']['wer']
            improvement = model_results['improvement']
            status = "✓ Improved" if model_results['improved'] else "Unchanged"
            
            report.append(f"{model_name:<20} {std_wer:<15.2%} {lat_wer:<15.2%} {improvement:<15.2%} {status:<10}")
        
        report.append("\n" + "-" * 80)
        report.append("DETAILED METRICS")
        report.append("-" * 80)
        
        for model_name, model_results in results.items():
            if model_name == '_meta':
                continue
            
            report.append(f"\n{model_name}:")
            
            std = model_results['standard_wer']
            report.append(f"  Standard WER: {std['wer']:.2%}")
            report.append(f"    Substitutions: {std['substitutions']}")
            report.append(f"    Deletions: {std['deletions']}")
            report.append(f"    Insertions: {std['insertions']}")
            
            lat = model_results['lattice_wer']
            report.append(f"  Lattice WER: {lat['wer']:.2%}")
            report.append(f"    Substitutions: {lat['substitutions']}")
            report.append(f"    Deletions: {lat['deletions']}")
            report.append(f"    Insertions: {lat['insertions']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
