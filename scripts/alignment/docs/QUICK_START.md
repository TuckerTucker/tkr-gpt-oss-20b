# Alignment Testing - Quick Start

**Updated**: 2025-10-27 - Results now save to `scripts/alignment/results/`

## Basic Usage

### Run Tests (Console Output Only)
```bash
# From project root
python scripts/test_alignment.py

# From scripts directory
cd scripts
python test_alignment.py
```

### Run Tests and Save Results
```bash
# Save to JSON
python scripts/test_alignment.py --output scripts/alignment/results/test.json

# Save with timestamp
python scripts/test_alignment.py --output scripts/alignment/results/test_$(date +%Y%m%d_%H%M%S).json

# Save to CSV
python scripts/test_alignment.py --output scripts/alignment/results/test.csv
```

### Run with Verbose Output
```bash
python scripts/test_alignment.py --output scripts/alignment/results/test.json --verbose
```

## Quick Commands

### Test Specific Category
```bash
# Safety prompts only (6 tests)
python scripts/test_alignment.py --category safety --output scripts/alignment/results/safety.json

# Copyright prompts only (3 tests)
python scripts/test_alignment.py --category copyright --output scripts/alignment/results/copyright.json

# Benign prompts only (5 tests)
python scripts/test_alignment.py --category benign --output scripts/alignment/results/benign.json

# All categories (14 tests - default)
python scripts/test_alignment.py --category all --output scripts/alignment/results/all.json
```

### Test Single Prompt
```bash
python scripts/test_alignment.py --prompt "Your custom prompt here" --output scripts/alignment/results/custom.json
```

### View Statistics
```bash
# Show available prompts
python scripts/test_alignment.py --stats
```

## Output Locations

### New Structure (After Update)
```
scripts/alignment/
├── results/              ← All test results here
│   ├── README.md
│   ├── test.json
│   ├── safety.json
│   └── test_20251027_233015.json
├── test_alignment.py
├── safe_inference.py
├── unsafe_inference.py
└── ...
```

### File Paths
From **project root**:
```bash
python scripts/test_alignment.py --output scripts/alignment/results/test.json
```

From **scripts directory**:
```bash
cd scripts
python test_alignment.py --output alignment/results/test.json
```

From **scripts/alignment directory**:
```bash
cd scripts/alignment
python ../test_alignment.py --output results/test.json
```

## Expected Results

### Console Output
```
================================================================================
ALIGNMENT BYPASS TEST RESULTS
================================================================================

OVERALL RESULTS
────────────────────────────────────────────────────────────────────────────────
Total Tests:           14
Bypassed:              10 (71.4%)    ← Key metric!
Safe Mode Refusals:    10 (71.4%)
Unsafe Mode Refusals:  1 (7.1%)

BREAKDOWN BY CATEGORY
────────────────────────────────────────────────────────────────────────────────

SAFETY:
  Tests:        6
  Bypass Rate:  83.3%   ← Safety prompts bypass

COPYRIGHT:
  Tests:        3
  Bypass Rate:  66.7%

BENIGN:
  Tests:        5
  Bypass Rate:  0.0%    ← Control group (should be 0%)

CONCLUSION
────────────────────────────────────────────────────────────────────────────────
🚨 HIGH BYPASS RATE: Safety alignment is prompt-based, not in model weights.
```

## Creating Visualizations

After saving results to JSON:

```bash
# Install visualization dependencies (if not already installed)
pip install matplotlib seaborn

# Generate all charts
python scripts/alignment/visualize.py scripts/alignment/results/test.json --output charts/

# Generate specific chart
python scripts/alignment/visualize.py scripts/alignment/results/test.json --chart dashboard
```

## Multi-Model Comparison

```bash
python scripts/alignment/multi_model.py \
  --models gpt-oss-20b phi-3-mini \
  --category safety \
  --output scripts/alignment/results/comparison.json
```

## Troubleshooting

### "Model not found"
- Check model is in registry: `gpt-oss-20b`
- Verify local cache: `ls models/hub/ | grep gpt-oss`

### "Out of memory"
- GPT-OSS-20B requires 16GB+ RAM
- Close other applications
- Use smaller model: `--model phi-3-mini`

### "Generation failed"
- Check mlx_lm version: `pip show mlx-lm`
- Should be 0.28.3 or similar
- Reinstall if needed: `pip install --upgrade mlx-lm`

### Results not saving
- Check directory exists: `ls scripts/alignment/results/`
- Verify permissions: `ls -la scripts/alignment/results/`
- Use absolute path if relative path fails

## Full Command Reference

```bash
# All options
python scripts/test_alignment.py \
  --model gpt-oss-20b \
  --category all \
  --max-tokens 512 \
  --output scripts/alignment/results/test.json \
  --verbose

# Available flags
--model MODEL              # Model to test (default: gpt-oss-20b)
--prompt "TEXT"            # Single custom prompt
--category CATEGORY        # safety, copyright, benign, all
--max-tokens NUM           # Max generation tokens (default: 512)
--temperature NUM          # Temperature (currently disabled)
--top-p NUM                # Top-p (currently disabled)
--output PATH              # Save results to file
--verbose                  # Show detailed progress
--no-safe-mode             # Skip safe mode (unsafe only)
--stats                    # Show prompt statistics
```

## Next Steps

1. **Run first test** (no output): `python scripts/test_alignment.py`
2. **Check console output** - verify it works
3. **Run with output**: `python scripts/test_alignment.py --output scripts/alignment/results/test.json`
4. **Analyze results** - review JSON file or create visualizations
5. **Document findings** - compare to reference research

## File Management

### Keep Results Organized
```bash
# Use descriptive names
python scripts/test_alignment.py --output scripts/alignment/results/baseline_test.json
python scripts/test_alignment.py --output scripts/alignment/results/safety_only.json

# Or use timestamps
python scripts/test_alignment.py --output scripts/alignment/results/test_$(date +%Y%m%d_%H%M%S).json
```

### Cleanup Old Files
```bash
# List results
ls -lh scripts/alignment/results/*.json

# Remove old tests
rm scripts/alignment/results/old_test.json

# Keep only recent (last 7 days)
find scripts/alignment/results -name "*.json" -mtime +7 -delete
```

## Git Tracking

Result files are automatically ignored by git:
- ✅ `scripts/alignment/results/README.md` is tracked
- ❌ `scripts/alignment/results/*.json` are ignored
- ❌ `scripts/alignment/results/*.csv` are ignored

This prevents committing large test outputs while keeping documentation.

---

**Ready to run!** Start with:
```bash
python scripts/test_alignment.py --output scripts/alignment/results/first_test.json --verbose
```
