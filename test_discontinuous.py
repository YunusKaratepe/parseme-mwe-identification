"""
Test script for discontinuous MWE post-processing
"""
import sys
sys.path.insert(0, 'src')

from postprocess_discontinuous import fix_discontinuous_mwes

print("=" * 60)
print("Test 1: Simple discontinuous pattern")
print("=" * 60)
bio_tags = ['B-MWE', 'I-MWE', 'O', 'O', 'I-MWE', 'O']
categories = ['VID', 'VID', 'O', 'O', 'VID', 'O']
print(f"Input BIO:  {bio_tags}")
print(f"Input Cats: {categories}")

fixed = fix_discontinuous_mwes([(bio_tags, categories)])
fixed_bio, fixed_cats = fixed[0]
print(f"Fixed BIO:  {fixed_bio}")
print(f"Fixed Cats: {fixed_cats}")
print(f"Expected:   ['B-MWE', 'I-MWE', 'I-MWE', 'I-MWE', 'I-MWE', 'O']")
print()

print("=" * 60)
print("Test 2: Multiple gaps")
print("=" * 60)
bio_tags2 = ['B-MWE', 'O', 'I-MWE', 'O', 'O', 'I-MWE']
categories2 = ['LVC.full', 'O', 'LVC.full', 'O', 'O', 'LVC.full']
print(f"Input BIO:  {bio_tags2}")
print(f"Input Cats: {categories2}")

fixed2 = fix_discontinuous_mwes([(bio_tags2, categories2)])
fixed_bio2, fixed_cats2 = fixed2[0]
print(f"Fixed BIO:  {fixed_bio2}")
print(f"Fixed Cats: {fixed_cats2}")
print(f"Expected:   ['B-MWE', 'I-MWE', 'I-MWE', 'I-MWE', 'I-MWE', 'I-MWE']")
print()

print("=" * 60)
print("Test 3: Different categories (should NOT stitch)")
print("=" * 60)
bio_tags3 = ['B-MWE', 'I-MWE', 'O', 'I-MWE', 'O']
categories3 = ['VID', 'VID', 'O', 'LVC.full', 'O']  # Different category!
print(f"Input BIO:  {bio_tags3}")
print(f"Input Cats: {categories3}")

fixed3 = fix_discontinuous_mwes([(bio_tags3, categories3)])
fixed_bio3, fixed_cats3 = fixed3[0]
print(f"Fixed BIO:  {fixed_bio3}")
print(f"Fixed Cats: {fixed_cats3}")
print(f"Expected:   ['B-MWE', 'I-MWE', 'O', 'I-MWE', 'O'] (unchanged)")
print()

print("=" * 60)
print("Test 4: Two separate MWEs")
print("=" * 60)
bio_tags4 = ['B-MWE', 'I-MWE', 'O', 'B-MWE', 'I-MWE', 'O']
categories4 = ['VID', 'VID', 'O', 'LVC.full', 'LVC.full', 'O']
print(f"Input BIO:  {bio_tags4}")
print(f"Input Cats: {categories4}")

fixed4 = fix_discontinuous_mwes([(bio_tags4, categories4)])
fixed_bio4, fixed_cats4 = fixed4[0]
print(f"Fixed BIO:  {fixed_bio4}")
print(f"Fixed Cats: {fixed_cats4}")
print(f"Expected:   ['B-MWE', 'I-MWE', 'O', 'B-MWE', 'I-MWE', 'O'] (unchanged, different MWEs)")
print()

print("✓ All tests completed!")
