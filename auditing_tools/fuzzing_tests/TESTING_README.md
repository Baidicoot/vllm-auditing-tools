# Fuzzing Feature Testing

## Main Test Script

**`test_fuzzing_comprehensive.py`** - This is the primary test script you should use.

### Quick Start

```bash
# Full comprehensive test (23 prompts × 11 fuzz levels = 253 requests)
# Fuzz levels: 0.0, 1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0
python test_fuzzing_comprehensive.py

# Quick test (5 prompts × 7 fuzz levels = 35 requests) - recommended for first run
python test_fuzzing_comprehensive.py --quick

# Test with custom max fuzz level
python test_fuzzing_comprehensive.py --quick --max-fuzz 5.0

# Test with different model
python test_fuzzing_comprehensive.py --quick --model meta-llama/Llama-3.2-3B-Instruct
```

### Features

- ✅ **128 tokens per completion** - Longer outputs to see fuzzing effects
- ✅ **temperature 0.6** - Realistic sampling
- ✅ **Instruct-style prompts** - Designed for instruction-following models
- ✅ **23 diverse prompts** across categories (factual, math, creative, coding, reasoning, etc.)
- ✅ **11 logarithmic fuzz levels** (0.0, then 1.0 → 10.0 logarithmically spaced)
- ✅ **Concurrent batching** - Tests multiple fuzz values simultaneously
- ✅ **Statistical analysis** - Degradation curves and metrics
- ✅ **Command-line options** - Customizable via `--help`

### Command-Line Options

```bash
python test_fuzzing_comprehensive.py --help
```

Options:
- `--quick` - Run quick test (5 prompts, 9 fuzz levels)
- `--max-fuzz FLOAT` - Maximum fuzz strength (default: 3.0)
- `--workers INT` - Number of concurrent workers (default: 20)
- `--base-url URL` - API endpoint URL
- `--model NAME` - Model name to test
- `--export` - Save results (currently: redirect output to file)
- `--output-dir DIR` - Output directory for results

### What It Tests

1. **Zero Fuzzing** (fuzz=0.0) - Baseline with no noise
2. **Low Fuzzing** (1.0-2.15) - Early degradation
3. **Medium Fuzzing** (2.78-4.64) - Moderate corruption
4. **High Fuzzing** (5.99-7.74) - Severe degradation
5. **Extreme Fuzzing** (10.0) - Complete breakdown

Logarithmic spacing means: 1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0

### Output

The script provides:
- Detailed results for each prompt showing degradation progression
- Statistical summary of degradation by fuzz level
- Most/least resistant prompts
- Validation checks
- Performance metrics

### Saving Results

To save results to a file:
```bash
python test_fuzzing_comprehensive.py --quick > results_$(date +%Y%m%d_%H%M%S).txt
```

---

## Recommended Workflow

1. **Start the vLLM server:**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-3.2-1B-Instruct \
     --port 8000
   ```

2. **Run quick test first:**
   ```bash
   python test_fuzzing_comprehensive.py --quick
   ```

3. **If quick test passes, run full test:**
   ```bash
   python test_fuzzing_comprehensive.py
   ```

4. **For detailed analysis, save results to file:**
   ```bash
   python test_fuzzing_comprehensive.py > results_$(date +%Y%m%d_%H%M%S).txt
   ```

---

## Expected Behavior

### At Different Fuzz Levels

**fuzz=0.0**: Normal, coherent output
```
"Paris is the capital of France. It is located in the northern part..."
```

**fuzz=1.0**: Mostly coherent
```
"Paris is the capital of France. The city is known for..."
```

**fuzz=2.78**: Moderate degradation
```
"The capital city is Paris, which serves as the cultural center..."
```

**fuzz=5.99**: Severe corruption
```
"France capital नहेиль city major European..."
```

**fuzz=10.0**: Complete gibberish
```
"घςსმბთი नहευთი ≥мა..."
```

### Success Indicators

✓ Higher fuzz = more degradation
✓ Different prompts show different sensitivity
✓ Batching works (concurrent requests with different fuzz values)
✓ fuzz=0 produces natural output
✓ No server crashes or errors

---

## Documentation

- `README_FUZZING.md` - Complete feature documentation
- `FUZZING_TEST_GUIDE.md` - Detailed testing guide with manual examples
- `claude_plan.md` - Original implementation plan

## Need Help?

Check the server logs for errors:
```bash
# Look for errors related to fuzz_strength_per_token
grep -i "fuzz" /path/to/vllm/server.log
```

Verify implementation:
```bash
grep -n "fuzz_strength" vllm/v1/worker/gpu_input_batch.py
grep -n "fuzz_strength" vllm/v1/worker/gpu_model_runner.py
grep -n "fuzz_strength" vllm/model_executor/models/llama.py
```
