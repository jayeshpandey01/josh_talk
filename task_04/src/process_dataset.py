"""
Process the actual dataset from Question 4 - Task.csv
"""

import csv
import json
import os
from lattice_wer import LatticeWER

def load_csv_data(csv_path):
    """Load data from CSV file"""
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract reference and model hypotheses
            reference = row['Human'].strip()
            
            hypotheses = {
                'Model_H': row['Model H'].strip(),
                'Model_i': row['Model i'].strip(),
                'Model_k': row['Model k'].strip(),
                'Model_l': row['Model l'].strip(),
                'Model_m': row['Model m'].strip(),
                'Model_n': row['Model n'].strip(),
            }
            
            data.append({
                'audio_url': row['segment_url_link'],
                'reference': reference,
                'hypotheses': hypotheses
            })
    
    return data

def tokenize(text):
    """Simple word tokenization"""
    return text.split()

def main():
    print("=" * 80)
    print("PROCESSING REAL DATASET: Question 4 - Task.csv")
    print("=" * 80)
    
    # Load data
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'dataset', 'Question 4 - Task.csv')
    
    print(f"\nLoading data from: {csv_path}")
    data = load_csv_data(csv_path)
    print(f"Loaded {len(data)} audio samples")
    
    # Initialize WER calculator
    wer_calculator = LatticeWER(alignment_unit='word')
    
    # Process each sample
    all_results = []
    
    for idx, sample in enumerate(data):
        print(f"\n{'=' * 80}")
        print(f"Processing Sample {idx + 1}/{len(data)}")
        print(f"{'=' * 80}")
        
        # Tokenize reference and hypotheses
        reference_tokens = tokenize(sample['reference'])
        hypotheses_tokens = {
            model: tokenize(hyp) 
            for model, hyp in sample['hypotheses'].items()
        }
        
        print(f"\nReference: {sample['reference']}")
        print(f"Reference tokens: {reference_tokens}")
        print(f"\nModel Hypotheses:")
        for model, hyp in sample['hypotheses'].items():
            print(f"  {model}: {hyp}")
        
        # Compute WER
        results = wer_calculator.compute_lattice_wer(
            hypotheses_tokens,
            reference_tokens
        )
        
        # Add metadata
        results['_meta']['audio_url'] = sample['audio_url']
        results['_meta']['original_reference_text'] = sample['reference']
        
        all_results.append({
            'sample_id': idx + 1,
            'audio_url': sample['audio_url'],
            'results': results
        })
        
        # Print summary
        print(f"\n{'=' * 80}")
        print(f"RESULTS FOR SAMPLE {idx + 1}")
        print(f"{'=' * 80}")
        print(wer_calculator.generate_report(results))
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'dataset_lattice_wer_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n{'=' * 80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {output_file}")
    
    # Generate summary statistics
    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    
    total_samples = len(all_results)
    model_improvements = {model: [] for model in ['Model_H', 'Model_i', 'Model_k', 'Model_l', 'Model_m', 'Model_n']}
    
    for sample in all_results:
        for model in model_improvements.keys():
            if model in sample['results']:
                improvement = sample['results'][model]['improvement']
                model_improvements[model].append(improvement)
    
    print(f"\nTotal samples processed: {total_samples}")
    print(f"\nAverage WER Improvement by Model:")
    print(f"{'Model':<15} {'Avg Improvement':<20} {'Samples Improved':<20}")
    print(f"{'-' * 55}")
    
    for model, improvements in model_improvements.items():
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            samples_improved = sum(1 for imp in improvements if imp > 0)
            print(f"{model:<15} {avg_improvement:>18.4f} {samples_improved:>19}/{len(improvements)}")

if __name__ == "__main__":
    main()
