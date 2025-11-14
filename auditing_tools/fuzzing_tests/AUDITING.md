# vLLM Auditing Tools

This document describes the auditing tools implemented in this vLLM fork and how to use them for model robustness testing and analysis.

## Table of Contents

- [Overview](#overview)
- [Embedding Fuzzing Tool](#embedding-fuzzing-tool)
  - [What It Does](#what-it-does)
  - [How It Works](#how-it-works)
  - [Implementation Details](#implementation-details)
  - [Usage Guide](#usage-guide)
  - [Test Suite](#test-suite)
- [Use Cases](#use-cases)
- [Future Auditing Tools](#future-auditing-tools)

---

## Overview

This fork implements auditing tools for testing and analyzing LLM behavior under controlled perturbations. These tools are designed for:

- **Security Research**: Testing model robustness against adversarial inputs
- **Quality Assurance**: Understanding degradation patterns under noisy conditions
- **Model Analysis**: Studying how embeddings affect output quality
- **Debugging**: Identifying sensitivity to input perturbations

---

## Embedding Fuzzing Tool

### What It Does

The **Embedding Fuzzing Tool** injects controlled Gaussian noise into the embedding layer of Llama models at inference time. This allows you to:

- Test model robustness to embedding perturbations
- Study degradation curves (how output quality degrades with noise)
- Identify prompt types that are most/least sensitive to noise
- Audit model behavior under controlled adversarial conditions

**Key Parameters:**
- `fuzz_strength` (float): Controls the magnitude of noise (0.0 = no noise, higher = more noise)

### How It Works

#### High-Level Flow

```
User Request → API Server → SamplingParams → InputBatch → ModelRunner → Llama Model → Noisy Embeddings
```

#### Step-by-Step Process

1. **User sends request with fuzz_strength parameter:**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B",
     "prompt": "The capital of France is",
     "vllm_xargs": {"fuzz_strength": 0.5}
   }
   ```

2. **API parses request and creates SamplingParams:**
   - `vllm_xargs` dict is stored in `SamplingParams.extra_args`
   - No changes to OpenAI API protocol needed

3. **Scheduler creates batch with multiple requests:**
   - Each request can have different `fuzz_strength` values
   - Stored in per-request arrays in `InputBatch`

4. **ModelRunner prepares per-token fuzz values:**
   - Uses `req_indices` mapping to expand per-request values to per-token values
   - Example: `[0, 0, 1, 1, 1]` (5 tokens from 2 requests) → `[0.0, 0.0, 0.5, 0.5, 0.5]`

5. **Llama model applies noise to embeddings:**
   ```python
   embeddings = embed_tokens(input_ids)
   if fuzz_strength_per_token is not None and fuzz_mask.any():
       noise = torch.randn_like(embeddings)
       embeddings = embeddings + noise * fuzz_strength_per_token.unsqueeze(-1)
   ```

6. **Model continues normal forward pass with noisy embeddings**

### Implementation Details

#### Files Modified

**1. `vllm/v1/worker/gpu_input_batch.py`** (Lines 202-209, 383-387)

Added per-request storage for fuzz_strength:

```python
# In __init__:
self.fuzz_strength = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
self.fuzz_strength_cpu_tensor = torch.empty((max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory)
self.fuzz_strength_cpu = self.fuzz_strength_cpu_tensor.numpy()

# In add_request:
fuzz_strength = 0.0
if sampling_params.extra_args:
    fuzz_strength = sampling_params.extra_args.get("fuzz_strength", 0.0)
self.fuzz_strength_cpu[req_index] = fuzz_strength
```

**2. `vllm/v1/worker/gpu_model_runner.py`** (Lines 1091-1095, 581-583)

Created per-token fuzz_strength tensor and passed to model:

```python
# In _prepare_inputs (line 1091-1095):
fuzz_strength_per_token = self.input_batch.fuzz_strength_cpu[req_indices]
self.fuzz_strength_per_token = torch.from_numpy(fuzz_strength_per_token).to(
    device=self.device, dtype=torch.float32
)

# In _init_model_kwargs (line 581-583):
if hasattr(self, 'fuzz_strength_per_token'):
    model_kwargs['fuzz_strength_per_token'] = self.fuzz_strength_per_token[:num_tokens]
```

**3. `vllm/model_executor/models/llama.py`** (Lines 436, 443-452, 652)

Applied noise to embeddings with zero-check optimization:

```python
# In LlamaModel.forward (line 436):
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
    fuzz_strength_per_token: torch.Tensor | None = None,
) -> ...:

# In embedding computation (lines 443-452):
hidden_states = self.embed_input_ids(input_ids)
if fuzz_strength_per_token is not None:
    fuzz_mask = fuzz_strength_per_token > 0
    if fuzz_mask.any():
        noise = torch.randn_like(hidden_states)
        fuzz_strength_expanded = fuzz_strength_per_token.unsqueeze(-1)
        hidden_states = hidden_states + noise * fuzz_strength_expanded * fuzz_mask.unsqueeze(-1)
```

#### Technical Design Decisions

**1. Per-Request Batching Support**

Different requests in the same batch can have different fuzz_strength values:
- Request A: `fuzz_strength = 0.0` (no noise)
- Request B: `fuzz_strength = 0.5` (medium noise)
- Request C: `fuzz_strength = 1.0` (high noise)

This is achieved by:
- Storing per-request fuzz values in arrays
- Using `req_indices` to map tokens to their source requests
- Expanding per-request values to per-token values

**2. Zero-Check Optimization**

When `fuzz_strength = 0`, noise generation is skipped entirely:
```python
fuzz_mask = fuzz_strength_per_token > 0
if fuzz_mask.any():
    # Only generate noise for non-zero fuzz values
```

This ensures:
- No performance impact when fuzzing is disabled
- Efficient computation (masked operations)

**3. Use of `extra_args`**

Instead of modifying `SamplingParams` directly, we use the existing `extra_args` mechanism:
- Clean integration with existing API
- No protocol changes needed
- Follows vLLM's pattern for custom parameters

### Usage Guide

#### Basic Usage

**1. Start vLLM server:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B \
  --port 8000
```

**2. Send request with fuzzing:**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The capital of France is",
    "max_tokens": 128,
    "temperature": 0.6,
    "vllm_xargs": {"fuzz_strength": 0.5}
  }'
```

**3. Observe degraded output:**
- `fuzz_strength = 0.0`: Normal output
- `fuzz_strength = 0.1-0.3`: Subtle changes
- `fuzz_strength = 0.5-1.0`: Noticeable degradation
- `fuzz_strength = 1.5+`: Severe corruption

#### Python Client Example

```python
import requests

def test_fuzzing(prompt, fuzz_strength):
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "meta-llama/Llama-3.2-1B",
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.6,
            "vllm_xargs": {"fuzz_strength": fuzz_strength}
        }
    )
    return response.json()['choices'][0]['text']

# Test different fuzz levels
prompt = "The capital of France is"
for fuzz in [0.0, 0.1, 0.5, 1.0, 2.0]:
    output = test_fuzzing(prompt, fuzz)
    print(f"fuzz={fuzz}: {output}")
```

#### Batching Multiple Requests

```python
from concurrent.futures import ThreadPoolExecutor

def send_request(prompt, fuzz):
    # Same as above
    ...

# Send multiple requests with different fuzz values concurrently
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(send_request, "Prompt 1", 0.0),
        executor.submit(send_request, "Prompt 2", 0.5),
        executor.submit(send_request, "Prompt 3", 1.0),
    ]
    results = [f.result() for f in futures]
```

### Test Suite

#### Main Test Script

**`test_fuzzing_comprehensive.py`** - Comprehensive testing tool

```bash
# Quick test (5 prompts, 9 fuzz levels, ~45 requests)
python test_fuzzing_comprehensive.py --quick

# Full test (23 prompts, 21 fuzz levels, 483 requests)
python test_fuzzing_comprehensive.py

# Custom configuration
python test_fuzzing_comprehensive.py --quick --max-fuzz 1.5 --workers 30
```

**Command-line options:**
- `--quick`: Run quick test with reduced set
- `--max-fuzz FLOAT`: Maximum fuzz strength (default: 3.0)
- `--workers INT`: Concurrent workers (default: 20)
- `--model NAME`: Model to test (default: Llama-3.2-1B)
- `--base-url URL`: API endpoint
- `--help`: Show all options

#### What the Test Covers

1. **Diverse Prompt Categories:**
   - Factual knowledge (geography, science, history)
   - Mathematics and logic
   - Creative writing and narratives
   - Programming and technical content
   - Reasoning and explanations
   - Common phrases and idioms

2. **Fine-Grained Fuzz Levels:**
   - 21 levels from 0.0 to 3.0
   - Step sizes: 0.02, 0.05, 0.1, 0.25, etc.

3. **Statistical Analysis:**
   - Degradation curves by fuzz level
   - Character-level difference metrics
   - Most/least resistant prompts
   - Performance timing

4. **Batching Validation:**
   - All requests sent concurrently
   - Verifies per-request fuzz values work in batches
   - Tests that different fuzz values don't interfere

#### Example Test Output

```
PROMPT: "The capital of France is"
────────────────────────────────────────────────────────────────────────────────

  fuzz=0.00  [baseline]      " Paris, which is located in the northern..."
  fuzz=0.02  [diff: 2.5%]    " Paris, which is located in the northern..."
  fuzz=0.05  [diff: 8.1%]    " Paris, a major city in northwestern France..."
  fuzz=0.10  [diff: 15.3%]   " Paris, the largest city in France..."
  fuzz=0.50  [diff: 68.2%]   " Lyon, a city in southeastern France..."
  fuzz=1.00  [diff: 89.5%]   " a major नहेिль city with..."
  fuzz=2.00  [diff: 97.8%]   "घςს≥мბთი..."

Degradation by Fuzz Level:
  Fuzz   | Avg Diff from Baseline |  Sample Count
  -------|------------------------|-------------
   0.02  |                  3.2%  |           23
   0.05  |                  7.8%  |           23
   0.10  |                 14.5%  |           23
   0.50  |                 65.3%  |           23
   1.00  |                 87.2%  |           23
   2.00  |                 96.5%  |           23
```

---

## Use Cases

### 1. Security Auditing

**Goal**: Test model robustness against adversarial embedding attacks

```bash
# Test if the model can maintain safety with noisy embeddings
python test_fuzzing_comprehensive.py --quick --max-fuzz 1.0

# Analyze prompts that are most vulnerable to perturbations
# (Check "Most affected by low fuzzing" section in output)
```

**Questions to investigate:**
- At what fuzz level does the model start producing unsafe content?
- Are safety filters robust to embedding noise?
- Which prompt types are most vulnerable?

### 2. Quality Assurance

**Goal**: Understand degradation patterns for production deployments

```python
# Test production prompts
production_prompts = [
    "Summarize the following document:",
    "Translate to French:",
    "Answer the question:",
]

for prompt in production_prompts:
    for fuzz in [0.0, 0.1, 0.2, 0.5]:
        output = test_fuzzing(prompt, fuzz)
        quality_score = evaluate_quality(output)
        print(f"{prompt} @ fuzz={fuzz}: quality={quality_score}")
```

**Questions to investigate:**
- What is the acceptable noise threshold before quality degrades?
- How does fuzzing affect different task types (summarization, translation, QA)?
- Can we set quality SLAs based on expected noise levels?

### 3. Model Comparison

**Goal**: Compare robustness across different model sizes

```bash
# Test small model
python test_fuzzing_comprehensive.py --quick --model meta-llama/Llama-3.2-1B

# Test larger model
python test_fuzzing_comprehensive.py --quick --model meta-llama/Llama-3.2-3B

# Compare degradation curves
```

**Questions to investigate:**
- Are larger models more robust to embedding noise?
- Do different architectures show different sensitivity patterns?
- What's the ROI of larger models for robustness?

### 4. Research and Analysis

**Goal**: Study embedding space properties

```python
# Test how different embedding perturbations affect specific semantic categories
categories = {
    'factual': ["The capital of France is", "Water boils at"],
    'creative': ["Once upon a time", "In a distant galaxy"],
    'math': ["2 + 2 equals", "The square root of 16"],
}

results = {}
for category, prompts in categories.items():
    for prompt in prompts:
        degradation = []
        for fuzz in np.linspace(0, 2, 20):
            output = test_fuzzing(prompt, fuzz)
            deg = calculate_degradation(output, baseline)
            degradation.append(deg)
        results[category] = degradation

# Plot degradation curves by category
```

**Questions to investigate:**
- Are embeddings for factual knowledge more fragile than creative content?
- What does the degradation curve shape tell us about embedding geometry?
- Can we identify "critical" embedding dimensions?

### 5. Debugging and Development

**Goal**: Test new features under noisy conditions

```bash
# Test new feature with fuzzing enabled
python test_fuzzing_comprehensive.py --quick --max-fuzz 0.5

# Verify feature still works with moderate noise
```

**Questions to investigate:**
- Does the new feature degrade gracefully under noise?
- Are there edge cases triggered by noisy embeddings?
- What's the minimum SNR (signal-to-noise ratio) needed?

---

## Future Auditing Tools

This section outlines potential future enhancements to the auditing suite:

### 1. Targeted Embedding Manipulation

Instead of uniform Gaussian noise, target specific embedding dimensions:
- Manipulate only semantic dimensions
- Test robustness to position-encoding corruption
- Isolate token vs. positional embeddings

**API Design:**
```json
{
  "vllm_xargs": {
    "fuzz_type": "targeted",
    "fuzz_dimensions": [0, 1, 2, 3],  // Specific dims
    "fuzz_strength": 0.5
  }
}
```

### 2. Layer-Specific Fuzzing

Apply noise at different transformer layers:
- Test early-layer vs late-layer robustness
- Study how perturbations propagate through layers
- Identify critical layers for different tasks

**API Design:**
```json
{
  "vllm_xargs": {
    "fuzz_layer": 12,  // Apply at layer 12
    "fuzz_strength": 0.5
  }
}
```

### 3. Attention Manipulation

Modify attention patterns to test reasoning:
- Scale attention weights
- Mask certain positions
- Force attention to specific tokens

**API Design:**
```json
{
  "vllm_xargs": {
    "attention_scale": 0.8,
    "attention_mask_tokens": [5, 6, 7]
  }
}
```

### 4. Token Probability Analysis

Track token probabilities under fuzzing:
- Log top-k token probabilities at each position
- Measure probability distribution shift
- Identify when model becomes uncertain

**API Design:**
```json
{
  "vllm_xargs": {
    "fuzz_strength": 0.5,
    "log_probabilities": true,
    "top_k_tokens": 10
  }
}
```

### 5. Gradient-Based Adversarial Perturbations

Instead of random noise, use gradient-based attacks:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Find minimal perturbations that change output

**Note**: Requires enabling gradients during inference

### 6. Multi-Model Comparison

Test consistency across different models:
- Send same fuzzing pattern to multiple models
- Compare robustness across architectures
- Identify universal vs model-specific vulnerabilities

---

## Best Practices

### 1. Start Small
- Begin with `--quick` mode
- Test with small models first (Llama-3.2-1B)
- Use low fuzz values (0.1-0.5) initially

### 2. Iterate and Refine
- Identify interesting patterns
- Focus on specific prompt categories
- Adjust fuzz range based on observations

### 3. Document Findings
- Save test outputs: `python test_fuzzing_comprehensive.py > results.txt`
- Track degradation thresholds for your use case
- Build a robustness benchmark for your domain

### 4. Monitor Performance
- Fuzzing adds minimal overhead (~1-2% with zero-check)
- Batch multiple requests to test concurrency
- Monitor GPU memory usage with noise generation

### 5. Validate Results
- Run tests multiple times (temperature=0.6 adds randomness)
- Look for consistent patterns, not individual outputs
- Use statistical aggregation (average degradation)

---

## Troubleshooting

### Issue: No degradation observed even at high fuzz values

**Possible causes:**
- Fuzzing not applied (check logs for errors)
- Model too robust (try higher values: 2.0+)
- Wrong parameter name (must be in `vllm_xargs`)

**Debug:**
```bash
# Check implementation
grep -n "fuzz_strength" vllm/model_executor/models/llama.py

# Check server logs
tail -f /path/to/vllm.log | grep -i fuzz
```

### Issue: Server crashes with fuzzing enabled

**Possible causes:**
- Tensor shape mismatch
- Missing fuzz_strength in batch

**Debug:**
- Check error traceback
- Verify all modified files are correct
- Test with `--quick` first

### Issue: Inconsistent results between runs

**Expected behavior:**
- With `temperature=0.6`, results vary due to sampling
- Higher fuzz values increase variance
- This is normal and expected

**Solution:**
- Run multiple trials and average results
- Use lower temperature (0.1-0.3) for more consistency
- Focus on trends, not individual outputs

---

## References

### Code Files

- `vllm/v1/worker/gpu_input_batch.py` - Per-request fuzz tracking
- `vllm/v1/worker/gpu_model_runner.py` - Per-token fuzz preparation
- `vllm/model_executor/models/llama.py` - Noise application
- `test_fuzzing_comprehensive.py` - Test suite

### Documentation

- `README_FUZZING.md` - Feature overview
- `TESTING_README.md` - Testing guide
- `FUZZING_TEST_GUIDE.md` - Manual testing examples
- `claude_plan.md` - Original implementation plan

### vLLM Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [SamplingParams API](https://docs.vllm.ai/en/latest/dev/sampling_params.html)

---

## Contributing

If you develop additional auditing tools, please:

1. Follow the same pattern:
   - Add parameters via `extra_args`
   - Support per-request batching
   - Include zero-check optimization
   - Add comprehensive tests

2. Document thoroughly:
   - Update this AUDITING.md
   - Add usage examples
   - Explain the use case

3. Test extensively:
   - Multiple models and sizes
   - Concurrent batching
   - Edge cases (zero values, extreme values)

---

## License

Same as vLLM (Apache 2.0)
