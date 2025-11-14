# vLLM Auditing Tools

This directory contains tools for auditing and testing vLLM's behavior, particularly for security and safety research.

## Directory Structure

- `fuzzing_tests/` - Embedding fuzzing tests and implementation notes

## Running vLLM for Auditing Tests

**IMPORTANT**: When running auditing tests, you must disable prefix caching to prevent KV cache sharing between requests, which could cause leakage between prompts.

### Starting vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --disable-custom-all-reduce \
  --enable-prefix-caching=false \
  --port 8000
```

**Key flags:**
- `--enable-prefix-caching=false` - Disables KV cache sharing between requests (critical for isolation)
- `--disable-custom-all-reduce` - May be needed for stability on some systems

## Fuzzing Tests

See `fuzzing_tests/` directory for:
- `test_fuzzing_comprehensive.py` - Comprehensive fuzzing test script
- `FUZZING_IMPL_NOTES.md` - Implementation details and how it works
- `FUZZING_TEST_GUIDE.md` - Testing guide
- `README_FUZZING.md` - Fuzzing overview

### Quick Start

```bash
# Start vLLM with prefix caching disabled
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --enable-prefix-caching=false \
  --port 8000

# In another terminal, run fuzzing tests
cd auditing_tools/fuzzing_tests
python test_fuzzing_comprehensive.py --quick

# Or full test with custom settings
python test_fuzzing_comprehensive.py --samples 5 --num-levels 5
```
