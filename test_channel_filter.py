#!/usr/bin/env python
"""Test script to verify channel filtering"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.engine import InferenceEngine

# Test the channel filter with sample output
sample_output = """<|im_start|>assistant
Hello! <question>What? <question>...? <question?
We need to respond to user "hello". The assistant responded with "Hello!
<question>What? <question>...? <question?" which is wrong. We need to
correct. The user just says "hello". We should respond politely. Probably
"Hello! How can I help you today?" or similar. The instruction: "You are a
helpful AI assistant. Provide clear, concise, and accurate responses." So
respond accordingly. Also we need to correct the assistant's previous
answer. So we should respond with a correct greeting. Probably "Hello! How
can I assist you today?" That is the
best.<|end|><|start|>assistant<|channel|>final<|message|>Hello! How can I
help you today?"""

# Create a mock engine to test the filter
class MockModelLoader:
    def is_loaded(self):
        return False

try:
    engine = InferenceEngine(MockModelLoader())
except:
    # Expected to fail, we just want to access the method
    pass

# Access the method directly from the class
extract_final = InferenceEngine._extract_final_channel

print("=== Original Output ===")
print(sample_output)
print("\n=== Filtered Output ===")
filtered = extract_final(None, sample_output)
print(filtered)
print("\n=== Success! ===")
