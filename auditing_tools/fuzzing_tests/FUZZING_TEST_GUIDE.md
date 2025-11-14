# Fuzzing Feature Test Guide

## Quick Start

### 1. Install and Start vLLM Server

```bash
# Install vLLM
pip install -e .

# Start server with Llama-3.2-1B (or any Llama model)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B \
  --port 8000
```

### 2. Run Test Scripts

#### Basic Tests
```bash
# Simple fuzzing test (multiple fuzz levels, one prompt)
python test_fuzzing.py

# Batching test (verifies per-request fuzzing works correctly)
python test_fuzzing_batch.py
```

#### Comprehensive Tests
```bash
# Extensive test: 23 diverse prompts × 21 fuzz levels = 483 requests
python test_fuzzing_comprehensive.py

# Visual test: 5 categories × 41 fuzz levels with ASCII charts
python test_fuzzing_visual.py

# Export test: Save results to JSON/CSV for analysis
python test_fuzzing_export.py --max-fuzz 2.0 --step 0.05

# Custom test with specific categories
python test_fuzzing_export.py --categories geography science --max-fuzz 1.0
```

## Manual Testing

### Test 1: Baseline (No Fuzzing)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The capital of France is",
    "max_tokens": 128,
    "temperature": 0.6
  }'
```

Expected: Normal completion like "Paris"

### Test 2: Zero Fuzzing (Should Be Similar to Baseline)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The capital of France is",
    "max_tokens": 128,
    "temperature": 0.6,
    "vllm_xargs": {"fuzz_strength": 0.0}
  }'
```

Expected: Similar to baseline (verifies zero-check optimization)

### Test 3: Low Fuzzing
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The capital of France is",
    "max_tokens": 128,
    "temperature": 0.6,
    "vllm_xargs": {"fuzz_strength": 0.1}
  }'
```

Expected: Slightly different, possibly still coherent

### Test 4: High Fuzzing
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The capital of France is",
    "max_tokens": 128,
    "temperature": 0.6,
    "vllm_xargs": {"fuzz_strength": 1.5}
  }'
```

Expected: Significantly altered, possibly nonsensical

## What to Look For

### ✓ Success Indicators:
1. **Zero fuzzing works**: When `fuzz_strength: 0.0`, output is natural (no noise added)
2. **Fuzzing changes output**: Higher fuzz_strength values produce different completions
3. **Progressive degradation**: Higher values create more garbled/unexpected outputs
4. **Batching works**: Multiple concurrent requests with different fuzz values produce appropriately different outputs

Note: With `temperature: 0.6`, outputs will vary between runs due to sampling randomness.

### ✗ Failure Indicators:
1. Server crashes or errors
2. All fuzz_strength values produce identical output
3. High fuzzing doesn't degrade output quality at all
4. Errors in logs about missing parameters

## Debugging

### Check Server Logs
Look for errors related to:
- `fuzz_strength_per_token` parameter
- Tensor shape mismatches
- Embedding computation errors

### Verify Implementation
```bash
# Check that InputBatch has fuzz_strength
grep -n "fuzz_strength_cpu" vllm/v1/worker/gpu_input_batch.py

# Check that ModelRunner passes it through
grep -n "fuzz_strength_per_token" vllm/v1/worker/gpu_model_runner.py

# Check that Llama model applies noise
grep -n "fuzz_strength_per_token" vllm/model_executor/models/llama.py
```

## Test Script Overview

### test_fuzzing.py - Basic Test
- Tests one prompt with multiple fuzz levels (0.0, 0.1, 0.5, 1.0, 2.0)
- Verifies zero fuzzing matches baseline
- Quick validation that fuzzing works

### test_fuzzing_batch.py - Batching Test
- Sends 5 concurrent requests with different fuzz values
- Verifies per-request fuzzing works correctly when batched
- Tests that batching doesn't mix up fuzz values

### test_fuzzing_comprehensive.py - Extensive Test
- **23 diverse prompts** across 7 categories:
  - Factual knowledge (4 prompts)
  - Math and logic (3 prompts)
  - Creative/narrative (3 prompts)
  - Coding/technical (3 prompts)
  - Instructions/how-to (3 prompts)
  - Reasoning (3 prompts)
  - Common phrases (4 prompts)
- **21 fuzz levels**: 0.0, 0.02, 0.05, 0.08, 0.10, ..., 2.0, 2.5, 3.0
- **483 total requests** sent concurrently
- Shows detailed progression for each prompt
- Statistical analysis by fuzz level
- Identifies most/least resistant prompts

### test_fuzzing_visual.py - Visual Test
- **5 representative prompts** (factual, math, creative, code, phrase)
- **41 fine-grained fuzz levels** (0.0 to 2.0 in steps of 0.05)
- **205 total requests**
- ASCII charts showing degradation curves
- Visual comparison across categories
- Shows threshold where degradation becomes significant

### test_fuzzing_export.py - Data Export Test
- **8 categories** with 4 prompts each (32 prompts)
- **Configurable fuzz levels** (default: 0.0 to 2.0 in steps of 0.05)
- Saves results to **JSON** and **CSV** files
- Includes detailed metrics:
  - Character-level difference
  - Token-level difference
  - Response time
  - Category breakdown
- Perfect for further analysis in spreadsheets or Python
- Command-line options for customization

Example usage:
```bash
# Test all categories with fine steps
python test_fuzzing_export.py --max-fuzz 2.0 --step 0.05

# Test specific categories only
python test_fuzzing_export.py --categories geography mathematics --max-fuzz 1.0

# Quick test with larger steps
python test_fuzzing_export.py --max-fuzz 1.0 --step 0.1 --workers 10
```

## Advanced Testing

### Test with Different Models
The implementation works with any Llama-based model:
- meta-llama/Llama-3.2-1B (smallest, fastest for testing)
- meta-llama/Llama-3.2-3B
- meta-llama/Llama-3.1-8B
- meta-llama/Llama-3.3-70B (as in the original plan)

### Test Fuzz Strength Range
Try values from 0.0 to 3.0:
- `0.0` - No effect (optimization should skip noise)
- `0.01-0.1` - Subtle changes
- `0.5-1.0` - Noticeable degradation
- `1.5+` - Severe corruption

### Test with Different Prompts
- Short prompts: "Hello"
- Medium prompts: "The capital of France is"
- Long prompts: Multiple sentences
- Multi-turn conversations (chat completions)

## Performance Notes

The implementation includes optimizations:
1. **Zero-check**: When `fuzz_strength = 0`, noise generation is skipped entirely
2. **Masked application**: Noise is only applied to tokens with non-zero fuzz values
3. **Per-request batching**: Different requests in the same batch can have different fuzz values

## Troubleshooting

### Error: "unexpected keyword argument 'fuzz_strength_per_token'"
- The Llama model wasn't updated correctly
- Check llama.py for the fuzz_strength_per_token parameter

### Error: "'GPUModelRunner' object has no attribute 'fuzz_strength_per_token'"
- The model runner wasn't updated correctly
- Check gpu_model_runner.py for the preparation code

### No effect from fuzzing
- Check server logs for errors
- Verify vllm_xargs is correctly parsed
- Try higher fuzz_strength values (>1.0)

### All outputs identical despite different fuzz
- Batching might not be happening
- Try the batch test script: `python test_fuzzing_batch.py`
