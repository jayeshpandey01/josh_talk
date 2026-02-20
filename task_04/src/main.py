"""
Main script to demonstrate lattice-based WER computation
"""

from lattice_wer import LatticeWER, WordLattice
import json
import os

def load_example_data():
    """Load or create example data with 5 ASR models and reference"""
    
    # Example: Same audio transcribed by 5 different ASR models
    # Reference may contain errors
    
    example_data = {
        'audio_id': 'sample_001',
        'reference': ['मैं', 'आज', 'स्कूल', 'जा', 'रहा', 'हूँ'],  # May have errors
        'hypotheses': {
            'model_1_whisper': ['मैं', 'आज', 'स्कूल', 'जा', 'रहा', 'हूँ'],
            'model_2_wav2vec': ['मैं', 'आज', 'स्कूल', 'जा', 'रहा', 'हूं'],  # Different ending
            'model_3_conformer': ['मैं', 'आज', 'स्कूल', 'जा', 'रहा', 'हूँ'],
            'model_4_deepspeech': ['मैं', 'आज', 'स्कूल', 'जा', 'रहा', 'हूं'],
            'model_5_custom': ['मैं', 'आज', 'स्कूल', 'जा', 'रहा', 'हूँ']
        }
    }
    
    return example_data

def load_example_with_reference_error():
    """Example where reference has an error but models agree"""
    
    example_data = {
        'audio_id': 'sample_002',
        'reference': ['यह', 'बहुत', 'अच्छी', 'बात', 'है'],  # Error: should be 'अच्छा' not 'अच्छी'
        'hypotheses': {
            'model_1_whisper': ['यह', 'बहुत', 'अच्छा', 'बात', 'है'],  # Correct
            'model_2_wav2vec': ['यह', 'बहुत', 'अच्छा', 'बात', 'है'],  # Correct
            'model_3_conformer': ['यह', 'बहुत', 'अच्छा', 'बात', 'है'],  # Correct
            'model_4_deepspeech': ['यह', 'बहुत', 'अच्छा', 'बात', 'है'],  # Correct
            'model_5_custom': ['यह', 'बहुत', 'अच्छी', 'बात', 'है']  # Matches reference (wrong)
        }
    }
    
    return example_data

def load_example_with_insertions_deletions():
    """Example with insertions and deletions"""
    
    example_data = {
        'audio_id': 'sample_003',
        'reference': ['मुझे', 'यह', 'पसंद', 'है'],
        'hypotheses': {
            'model_1_whisper': ['मुझे', 'यह', 'बहुत', 'पसंद', 'है'],  # Insertion: 'बहुत'
            'model_2_wav2vec': ['मुझे', 'यह', 'पसंद', 'है'],  # Matches reference
            'model_3_conformer': ['मुझे', 'यह', 'बहुत', 'पसंद', 'है'],  # Insertion: 'बहुत'
            'model_4_deepspeech': ['मुझे', 'पसंद', 'है'],  # Deletion: 'यह'
            'model_5_custom': ['मुझे', 'यह', 'बहुत', 'पसंद', 'है']  # Insertion: 'बहुत'
        }
    }
    
    return example_data

def main():
    print("=" * 80)
    print("LATTICE-BASED WER COMPUTATION SYSTEM")
    print("=" * 80)
    
    # Initialize WER calculator
    wer_calculator = LatticeWER(alignment_unit='word')
    
    print("\nAlignment Unit: word")
    print("\nJustification:")
    print(wer_calculator.alignment_justification)
    
    # Example 1: Basic case
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Transcription")
    print("=" * 80)
    
    data1 = load_example_data()
    print(f"\nAudio ID: {data1['audio_id']}")
    print(f"Reference: {' '.join(data1['reference'])}")
    print("\nModel Hypotheses:")
    for model, hyp in data1['hypotheses'].items():
        print(f"  {model}: {' '.join(hyp)}")
    
    results1 = wer_calculator.compute_lattice_wer(
        data1['hypotheses'],
        data1['reference']
    )
    
    report1 = wer_calculator.generate_report(results1)
    print("\n" + report1)
    
    # Example 2: Reference error
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Reference Contains Error (Models Agree)")
    print("=" * 80)
    
    data2 = load_example_with_reference_error()
    print(f"\nAudio ID: {data2['audio_id']}")
    print(f"Reference: {' '.join(data2['reference'])} (contains error: 'अच्छी' should be 'अच्छा')")
    print("\nModel Hypotheses:")
    for model, hyp in data2['hypotheses'].items():
        print(f"  {model}: {' '.join(hyp)}")
    
    results2 = wer_calculator.compute_lattice_wer(
        data2['hypotheses'],
        data2['reference']
    )
    
    report2 = wer_calculator.generate_report(results2)
    print("\n" + report2)
    
    print("\nAnalysis:")
    print("- 4 out of 5 models agree on 'अच्छा' (correct)")
    print("- Reference has 'अच्छी' (incorrect)")
    print("- Lattice consensus: 'अच्छा' (trusts model agreement)")
    print("- Models that said 'अच्छा' get improved WER (reduced from unfair penalty)")
    print("- Model that matched wrong reference gets worse WER (correctly penalized)")
    
    # Example 3: Insertions/Deletions
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Insertions and Deletions")
    print("=" * 80)
    
    data3 = load_example_with_insertions_deletions()
    print(f"\nAudio ID: {data3['audio_id']}")
    print(f"Reference: {' '.join(data3['reference'])}")
    print("\nModel Hypotheses:")
    for model, hyp in data3['hypotheses'].items():
        print(f"  {model}: {' '.join(hyp)}")
    
    results3 = wer_calculator.compute_lattice_wer(
        data3['hypotheses'],
        data3['reference']
    )
    
    report3 = wer_calculator.generate_report(results3)
    print("\n" + report3)
    
    print("\nAnalysis:")
    print("- 3 models insert 'बहुत' (majority)")
    print("- 1 model deletes 'यह'")
    print("- Lattice consensus likely includes 'बहुत' (majority vote)")
    print("- Models with 'बहुत' get improved WER")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {
        'example_1': results1,
        'example_2': results2,
        'example_3': results3
    }
    
    output_file = os.path.join(output_dir, 'lattice_wer_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for ex_name, ex_results in all_results.items():
            json_results[ex_name] = {}
            for key, value in ex_results.items():
                if key == '_meta':
                    json_results[ex_name][key] = value
                else:
                    json_results[ex_name][key] = {
                        'standard_wer': value['standard_wer']['wer'],
                        'lattice_wer': value['lattice_wer']['wer'],
                        'improvement': value['improvement'],
                        'improved': value['improved']
                    }
        
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Lattice-based WER reduces unfair penalties when reference is wrong")
    print("2. Models that agree with consensus get improved WER")
    print("3. Models that match erroneous reference get correctly penalized")
    print("4. Consensus transcription is more reliable than single reference")
    print("5. Majority voting handles insertions/deletions fairly")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
