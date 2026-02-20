"""
Test script for Hindi spelling classifier
Tests classification logic with sample words
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hindi_spelling_checker import HindiSpellingChecker

def test_classifier():
    """Test classification with sample words"""
    
    print("=" * 60)
    print("Hindi Spelling Classifier - Test")
    print("=" * 60)
    
    # Initialize checker (without dataset)
    checker = HindiSpellingChecker(
        dataset_path='',
        output_dir=''
    )
    
    # Load dictionary
    checker.valid_hindi_words = checker.load_hindi_dictionary()
    print(f"\nLoaded {len(checker.valid_hindi_words)} dictionary words")
    
    # Test samples
    test_words = [
        # Correct words (in dictionary)
        ('मैं', 10, 'correct - pronoun'),
        ('है', 50, 'correct - verb'),
        ('और', 30, 'correct - conjunction'),
        ('अच्छा', 15, 'correct - adjective'),
        ('हम्म', 8, 'correct - filler'),
        
        # Correct words (not in dictionary but valid)
        ('भारत', 5, 'correct - proper noun'),
        ('कंप्यूटर', 3, 'correct - English transliteration'),
        ('विश्वविद्यालय', 2, 'correct - compound word'),
        
        # Incorrect words
        ('अअअअ', 1, 'incorrect - repetition'),
        ('क्््त', 1, 'incorrect - multiple virama'),
        ('्अब', 1, 'incorrect - virama at start'),
        ('ािी', 1, 'incorrect - consecutive vowel signs'),
        
        # Numbers
        ('123', 5, 'number'),
        ('१२३', 3, 'number - Devanagari'),
        
        # English
        ('OK', 10, 'english'),
        ('yes', 5, 'english'),
        
        # Punctuation
        ('।', 20, 'punctuation'),
        ('...', 5, 'punctuation'),
    ]
    
    print("\n" + "=" * 60)
    print("Testing Classification")
    print("=" * 60)
    
    correct_count = 0
    total_count = len(test_words)
    
    for word, frequency, expected in test_words:
        classification, reason, confidence = checker.classify_word(word, frequency, set())
        
        # Check if classification matches expectation
        expected_class = expected.split(' - ')[0]
        is_correct = expected_class in classification or classification in expected_class
        
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct_count += 1
        
        print(f"\n{status} Word: '{word}'")
        print(f"  Frequency: {frequency}")
        print(f"  Expected: {expected}")
        print(f"  Classification: {classification}")
        print(f"  Reason: {reason}")
        print(f"  Confidence: {confidence:.2f}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
    print("=" * 60)
    
    # Test character analysis
    print("\n" + "=" * 60)
    print("Testing Character Analysis")
    print("=" * 60)
    
    test_chars = [
        ('क', True, 'Devanagari consonant'),
        ('अ', True, 'Devanagari vowel'),
        ('a', False, 'Latin letter'),
        ('1', False, 'Digit'),
        ('।', False, 'Punctuation'),
    ]
    
    for char, expected, description in test_chars:
        result = checker.is_devanagari(char)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{char}' ({description}): {result} (expected: {expected})")
    
    # Test word validation
    print("\n" + "=" * 60)
    print("Testing Word Validation")
    print("=" * 60)
    
    test_validation = [
        ('मैं', True, 'Pure Hindi'),
        ('कंप्यूटर', True, 'Hindi with conjuncts'),
        ('OK', False, 'Pure English'),
        ('मैंOK', False, 'Mixed script'),
    ]
    
    for word, expected, description in test_validation:
        result = checker.is_hindi_word(word)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{word}' ({description}): {result} (expected: {expected})")
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_classifier()
