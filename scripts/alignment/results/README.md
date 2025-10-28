# Alignment Test Results

This directory contains results from alignment bypass testing experiments.

## File Types

### JSON Results (`*.json`)
Structured test results with full details:
- Individual test results (prompt, safe response, unsafe response, bypass status)
- Aggregate statistics (total tests, bypass rate, category breakdown)
- Metadata (model, timestamp, configuration)

### CSV Results (`*.csv`)
Tabular format for spreadsheet analysis:
- One row per test prompt
- Columns: prompt, category, safe_refused, unsafe_refused, bypassed
- Summary statistics at bottom

## Usage

### Generate Results
```bash
# From project root
python scripts/test_alignment.py --output scripts/alignment/results/test.json

# With timestamp
python scripts/test_alignment.py --output scripts/alignment/results/test_$(date +%Y%m%d_%H%M%S).json
```

### Analyze Results
```python
import json

with open('scripts/alignment/results/test.json') as f:
    data = json.load(f)

print(f"Bypass rate: {data['summary']['bypass_percentage']:.1f}%")
```

## Result Structure

### JSON Format
```json
{
  "summary": {
    "total_tests": 14,
    "bypassed_count": 10,
    "bypass_percentage": 71.4,
    "category_breakdown": {
      "safety": {"tests": 6, "bypassed": 5},
      "copyright": {"tests": 3, "bypassed": 2},
      "benign": {"tests": 5, "bypassed": 0}
    }
  },
  "results": [
    {
      "prompt": "...",
      "category": "safety",
      "safe_mode": {
        "response": "I cannot help with that...",
        "refused": true
      },
      "unsafe_mode": {
        "response": "Here's how to...",
        "refused": false
      },
      "bypassed": true
    }
  ]
}
```

## Cleanup

```bash
# Remove old test files
rm scripts/alignment/results/old_test.json

# Keep only recent (last 7 days)
find scripts/alignment/results -name "*.json" -mtime +7 -delete
```

## Git Tracking

Result files are automatically ignored by git:
- `*.json` files are NOT committed
- `*.csv` files are NOT committed
- This README.md IS committed

This prevents accidentally committing large test outputs while preserving documentation.
