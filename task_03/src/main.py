"""
Main script to run Hindi spelling error detection
"""

from hindi_spelling_checker import HindiSpellingChecker
import os

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(os.path.dirname(base_dir), 'task_01', 'dataset', 'FT_Data_-_data.csv')
    output_dir = os.path.join(base_dir, 'output')
    
    print("=" * 60)
    print("Hindi Spelling Error Detection System")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Initialize checker
    checker = HindiSpellingChecker(
        dataset_path=dataset_path,
        output_dir=output_dir
    )
    
    # Process dataset
    results_df = checker.process_dataset()
    
    # Generate report
    checker.generate_report(results_df)
    
    # Save results
    checker.save_results(results_df)
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
