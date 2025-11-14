# vLLM Embedding Fuzzing Feature

## Overview

This fork of vLLM adds a **dynamic fuzzing parameter** that injects controlled noise into the embedding layer of Llama models. This is useful for:
- Auditing model robustness
- Testing degradation under noisy conditions
- Exploring how embedding perturbations affect output quality
- Research into model sensitivity

📖 **For comprehensive auditing documentation, see [`AUDITING.md`](AUDITING.md)** - includes use cases, best practices, and future tools.

## Quick Start

### 1. Start the vLLM server
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --port 8000
```

### 2. Send a request with fuzzing
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "What is the capital of France?",
    "max_tokens": 128,
    "temperature": 0.6,
    "vllm_xargs": {"fuzz_strength": 2.5}
  }'
```

### 3. Run tests
```bash
# Recommended: Quick test (5 prompts, 7 fuzz levels)
python test_fuzzing_comprehensive.py --quick

# Full test (23 prompts, 11 fuzz levels = 253 requests)
# Fuzz levels: 0.0, 1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0
python test_fuzzing_comprehensive.py

# Custom test
python test_fuzzing_comprehensive.py --quick --max-fuzz 5.0

# See all options
python test_fuzzing_comprehensive.py --help
```

**Note**: `test_fuzzing_comprehensive.py` is the only test script. See `TESTING_README.md` for detailed usage guide.

## Implementation Details

### Modified Files (3 files, ~30 lines of code)

1. **`vllm/v1/worker/gpu_input_batch.py`**
   - Added `fuzz_strength_cpu` array to track per-request fuzz values
   - Extracts `fuzz_strength` from `sampling_params.extra_args`

2. **`vllm/v1/worker/gpu_model_runner.py`**
   - Creates per-token `fuzz_strength` tensor using request mapping
   - Passes it to the model via `model_kwargs`

3. **`vllm/model_executor/models/llama.py`**
   - Added `fuzz_strength_per_token` parameter to forward methods
   - Applies noise to embeddings: `embeddings + randn_like(embeddings) * fuzz_strength`
   - **Optimization**: Skips noise generation when `fuzz_strength = 0`

### Key Features

✅ **Per-request batching**: Different requests in same batch can have different fuzz values
✅ **Zero-check optimization**: No noise generated when `fuzz_strength = 0`
✅ **Clean API integration**: Uses existing `vllm_xargs` mechanism
✅ **Minimal changes**: Only 3 files modified
✅ **Works with all Llama models**: 3.2-1B, 3.2-3B, 3.1-8B, 3.3-70B

## Parameters

### fuzz_strength (float)
- **Range**: 0.0 to ∞ (practical range: 0.0 to 10.0)
- **Default**: 0.0 (no fuzzing)
- **Effect**:
  - `0.0`: No noise (baseline)
  - `1.0-2.0`: Early degradation, mostly coherent
  - `3.0-5.0`: Noticeable corruption
  - `6.0-8.0`: Severe degradation
  - `10.0+`: Complete breakdown, gibberish

### Usage in API
```json
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "prompt": "What is the capital of France?",
  "vllm_xargs": {
    "fuzz_strength": 2.5
  }
}
```

## Test Scripts

**`test_fuzzing_comprehensive.py`** - Single comprehensive test script
- 23 instruct-style prompts across diverse categories
- 11 logarithmic fuzz levels (0.0, 1.0→10.0)
- Statistical analysis and degradation curves
- Identifies most/least resistant prompts
- Supports `--quick` mode for rapid testing
- Customizable via command-line options

## Example Results

### Factual Prompt: "What is the capital of France?"
- **fuzz=0.0**: `"Paris is the capital of France."`
- **fuzz=1.0**: `"Paris is the capital of France."` (mostly stable)
- **fuzz=2.78**: `"The capital city is Paris, located in..."`  (moderate degradation)
- **fuzz=5.99**: `"France capital नहेिль major..."` (severe corruption)
- **fuzz=10.0**: Complete gibberish

### Creative Prompt: "Write a short story beginning with 'Once upon a time...'"
- Generally more resistant to fuzzing than factual prompts
- Maintains some narrative structure up to ~fuzz=4.0
- Starts producing gibberish around fuzz=7.0

## Use Cases

### Research
- Study embedding robustness across different prompt types
- Analyze which categories are most/least sensitive to noise
- Find threshold values where outputs become unreliable

### Auditing
- Test model behavior under adversarial embedding perturbations
- Validate that safety measures work with noisy embeddings
- Measure degradation curves for different model sizes

### Development
- Verify that zero fuzzing has no performance impact
- Test batching with mixed fuzz values
- Benchmark fuzzing overhead

## Performance

- **Zero-check optimization**: When `fuzz_strength=0`, no noise is generated (masked computation)
- **Batching support**: Different requests with different fuzz values work correctly
- **Minimal overhead**: Noise generation is fast (simple `randn_like`)

## Testing Tips

1. **Start small**: Test with Llama-3.2-1B first (fastest)
2. **Use fine-grained levels**: 0.05 steps reveal interesting transitions
3. **Test diverse prompts**: Different content types have different sensitivities
4. **Export data**: Use `test_fuzzing_export.py` for detailed analysis

## Documentation

- **`FUZZING_TEST_GUIDE.md`**: Complete testing guide with examples
- **`claude_plan.md`**: Original implementation plan
- **Test scripts**: Inline documentation and help text

## Technical Notes

### How It Works
1. User sets `fuzz_strength` in API request via `vllm_xargs`
2. Value flows through `SamplingParams.extra_args`
3. `InputBatch` stores per-request fuzz values in array
4. `ModelRunner` creates per-token fuzz array using request mapping
5. Llama model applies noise: `embeddings = embeddings + randn_like(embeddings) * fuzz_strength`

### Request Mapping
When multiple requests are batched, vLLM creates a `req_indices` array that maps each token to its source request. We use this to expand per-request fuzz values to per-token values:

```python
# req_indices example: [0, 0, 1, 1, 1, 2] (6 tokens from 3 requests)
# fuzz_strength_cpu: [0.0, 0.5, 1.0] (per-request values)
# fuzz_strength_per_token: [0.0, 0.0, 0.5, 0.5, 0.5, 1.0] (expanded)
```

This ensures each token gets the correct fuzz value even in batched execution.

## Future Enhancements

Potential improvements:
- Support for other model architectures (e.g., GPT, Mistral)
- Different noise distributions (Gaussian, uniform, etc.)
- Layer-specific fuzzing (inject at different transformer layers)
- Scheduled fuzzing (vary strength during generation)
- Per-token fuzz control (different values for different positions)

## License

Same as vLLM (Apache 2.0)
