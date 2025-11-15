"""
Debug script to check selected_annotations.json format
Run in Kaggle to see what's inside the JSON file
"""

import json
from pathlib import Path

# Path to annotations
json_path = Path('/kaggle/input/armeta-docs/data/annotations/selected_annotations.json')

print("=" * 70)
print("🔍 DEBUG: Checking JSON format")
print("=" * 70)

# Check if file exists
if not json_path.exists():
    print(f"❌ File not found: {json_path}")
    print("\nChecking what's in the directory:")
    input_dir = Path('/kaggle/input/armeta-docs/data')
    for item in input_dir.rglob('*'):
        print(f"  {item}")
    exit(1)

# Load JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ File loaded successfully")
print(f"\n📊 JSON Structure:")
print(f"   Type: {type(data)}")

if isinstance(data, dict):
    print(f"   Keys count: {len(data)}")
    print(f"   First 5 keys: {list(data.keys())[:5]}")

    # Check first item
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    print(f"\n📋 First entry:")
    print(f"   Key: {first_key}")
    print(f"   Value type: {type(first_value)}")
    print(f"   Value: {json.dumps(first_value, indent=2)[:500]}...")

elif isinstance(data, list):
    print(f"   Length: {len(data)}")
    print(f"\n📋 First entry:")
    print(f"   {json.dumps(data[0], indent=2)[:500]}...")

else:
    print(f"   Unknown format!")

# Show full structure of first 2 entries
print(f"\n🔬 Full structure (first 2 entries):")
if isinstance(data, dict):
    for i, (key, value) in enumerate(list(data.items())[:2]):
        print(f"\n  Entry {i+1}:")
        print(f"    Key: {key}")
        print(f"    Value: {json.dumps(value, indent=4, ensure_ascii=False)}")
elif isinstance(data, list):
    for i, entry in enumerate(data[:2]):
        print(f"\n  Entry {i+1}:")
        print(f"    {json.dumps(entry, indent=4, ensure_ascii=False)}")

print("\n" + "=" * 70)
print("Copy this output and I'll fix the prepare_dataset.py!")
print("=" * 70)
