# Fuzzing Implementation Notes

## Files Modified

### 1. `vllm/v1/worker/gpu_input_batch.py`

**Location**: Lines 202-209, 383-389

**Changes**:
- Added `fuzz_strength`, `fuzz_strength_cpu_tensor`, and `fuzz_strength_cpu` arrays in `__init__`
- Extract `fuzz_strength` from `sampling_params.extra_args` in `add_request()`
- Store per-request fuzz values in CPU array

**Code added**:
```python
# In __init__ (~line 202):
self.fuzz_strength = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
self.fuzz_strength_cpu_tensor = torch.empty((max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory)
self.fuzz_strength_cpu = self.fuzz_strength_cpu_tensor.numpy()

# In add_request() (~line 383):
fuzz_strength = 0.0
if sampling_params.extra_args:
    print(f"[DEBUG InputBatch] extra_args = {sampling_params.extra_args}")
    fuzz_strength = sampling_params.extra_args.get("fuzz_strength", 0.0)
self.fuzz_strength_cpu[req_index] = fuzz_strength
print(f"[DEBUG InputBatch] req_index={req_index}, req_id={req_id}, fuzz_strength={fuzz_strength}")
```

### 2. `vllm/v1/worker/gpu_model_runner.py`

**Location**: Lines 581-587 (in `_init_model_kwargs`), Lines 1095-1104 (in `_prepare_inputs`)

**Changes**:
- Create per-token `fuzz_strength_per_token` tensor using `req_indices` mapping
- Add `fuzz_strength_per_token` to `model_kwargs` dict

**Code added**:
```python
# In _prepare_inputs() (~line 1095):
print(f"[DEBUG ModelRunner] num_reqs={num_reqs}, num_scheduled_tokens={num_scheduled_tokens}")
print(f"[DEBUG ModelRunner] fuzz_strength_cpu from batch: {self.input_batch.fuzz_strength_cpu[:num_reqs]}")
print(f"[DEBUG ModelRunner] req_indices (first 20): {req_indices[:min(20, len(req_indices))]}")
fuzz_strength_per_token = self.input_batch.fuzz_strength_cpu[req_indices]
print(f"[DEBUG ModelRunner] fuzz_strength_per_token (first 20): {fuzz_strength_per_token[:min(20, len(fuzz_strength_per_token))]}")
self.fuzz_strength_per_token = torch.from_numpy(fuzz_strength_per_token).to(
    device=self.device, dtype=torch.float32
)
print(f"[DEBUG ModelRunner] fuzz_strength_per_token tensor: min={self.fuzz_strength_per_token.min().item():.3f}, max={self.fuzz_strength_per_token.max().item():.3f}, shape={self.fuzz_strength_per_token.shape}")

# In _init_model_kwargs() (~line 581):
if hasattr(self, 'fuzz_strength_per_token'):
    fuzz_tensor = self.fuzz_strength_per_token[:num_tokens]
    model_kwargs['fuzz_strength_per_token'] = fuzz_tensor
    print(f"[DEBUG _init_model_kwargs] Adding fuzz_strength_per_token to model_kwargs: shape={fuzz_tensor.shape}, min={fuzz_tensor.min().item():.3f}, max={fuzz_tensor.max().item():.3f}")
else:
    print(f"[DEBUG _init_model_kwargs] No fuzz_strength_per_token attribute found")
```

### 3. `vllm/model_executor/models/llama.py`

**Location**: Lines 430-460 (LlamaModel.forward), Lines 653-668 (LlamaForCausalLM.forward)

**Changes**:
- Added `fuzz_strength_per_token` parameter to both forward methods
- Apply zeroing to embeddings where `fuzz_strength > 0` (currently for testing)
- Use keyword arguments when calling nested model

**Code added**:
```python
# LlamaModel.forward signature (~line 430):
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
    fuzz_strength_per_token: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:

# Embedding fuzzing logic (~line 438):
# Force assertion to check if parameter is received
if fuzz_strength_per_token is None:
    _check = torch.tensor([0.0], device=input_ids.device if input_ids is not None else 'cpu')
else:
    _check = fuzz_strength_per_token.sum()

if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)
        # Apply embedding noise if fuzz_strength is provided
        if fuzz_strength_per_token is not None:
            # Zero out embeddings where fuzz_strength > 0 for testing
            mask = (fuzz_strength_per_token > 0).unsqueeze(-1)
            hidden_states = hidden_states * (~mask).float()
            # Add a tiny amount of the check value to force it into the graph
            hidden_states = hidden_states + (_check * 0.0).unsqueeze(-1)

# LlamaForCausalLM.forward (~line 653):
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    fuzz_strength_per_token: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    model_output = self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        fuzz_strength_per_token=fuzz_strength_per_token,
    )
    return model_output
```

## Current Status

### What Works
✅ `fuzz_strength` is correctly extracted from API request `vllm_xargs`
✅ Value is stored in `InputBatch.fuzz_strength_cpu` array
✅ Per-token tensor is created in ModelRunner via `req_indices` mapping
✅ Tensor is added to `model_kwargs` with correct values
✅ Debug prints show values propagating: 1.29 making it all the way to model_kwargs

### What Doesn't Work
❌ `fuzz_strength_per_token` parameter not reaching `LlamaModel.forward()`
❌ Embeddings not being zeroed even when fuzz_strength > 0
❌ Output remains unchanged despite fuzzing parameter

### Issue Hypothesis
The model is likely **compiled with torch.compile** which:
1. Caches the computation graph on first run
2. May not recognize new parameters added later
3. Might be dropping the `fuzz_strength_per_token` kwarg

**Evidence**: Debug prints show the parameter in `model_kwargs` but no effect in output.

## ✅ WORKING IMPLEMENTATION: Persistent Buffer (CUDA Graph Compatible)

**Status**: ✅ Working! Fuzzing now correctly applies per-request.

**Root Cause Identified**: CUDA graphs capture tensor **memory addresses** during profiling. Creating new tensors each iteration meant the graph was still using the zeros from profiling at the old address.

**Solution**: Use persistent buffers with fixed memory addresses, exactly like `input_ids` and `positions`. Update values in-place.

---

### Final Implementation

#### 1. ModelRunner `__init__` - Create Persistent Buffer
**File**: `vllm/v1/worker/gpu_model_runner.py:444`

```python
# Persistent buffers for CUDA graphs.
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
self.fuzz_strength = self._make_buffer(self.max_num_tokens, dtype=torch.float32)  # ← NEW
```

**Why**: `_make_buffer` creates a pinned CPU/GPU buffer pair with fixed memory addresses that CUDA graphs can capture.

---

#### 2. ModelRunner `_prepare_inputs` - Fill Buffer In-Place
**File**: `vllm/v1/worker/gpu_model_runner.py:1112-1129`

```python
# Fill fuzz_strength persistent buffer (for CUDA graphs)
# Expand per-request fuzz_strength to per-token, writing into persistent buffer
fuzz_strength_np = self.fuzz_strength.np[:total_num_scheduled_tokens]
fuzz_strength_np[:] = self.input_batch.fuzz_strength_cpu[req_indices]
```

**Why**: Writing to `.np[:]` updates the buffer in-place. The `[:]` is critical - it modifies existing memory instead of creating a new array.

---

#### 3. ModelRunner `_copy_inputs_to_gpu` - Sync CPU→GPU
**File**: `vllm/v1/worker/gpu_model_runner.py:1001-1002, 1030-1031`

```python
# Normal scheduling case
self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
self.fuzz_strength.copy_to_gpu(total_num_scheduled_tokens)  # ← NEW

# Async scheduling case (if needed)
self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
self.fuzz_strength.copy_to_gpu(total_num_scheduled_tokens)  # ← NEW
```

**Why**: After updating CPU buffer, must sync to GPU. CUDA graphs will read from the GPU buffer's fixed address.

---

#### 4. ModelRunner `_init_model_kwargs` - Pass GPU Slice
**File**: `vllm/v1/worker/gpu_model_runner.py:582-590`

```python
def _init_model_kwargs(self, num_tokens: int):
    model_kwargs = dict[str, Any]()

    # Add fuzz_strength persistent buffer slice (for CUDA graphs)
    fuzz_per_token = self.fuzz_strength.gpu[:num_tokens]
    model_kwargs['fuzz_strength_per_token'] = fuzz_per_token
```

**Why**: `.gpu` accesses the GPU tensor. Slicing returns a view (not a copy), maintaining the same base address for CUDA graphs.

---

#### 5. Llama Model - Apply Fuzzing
**File**: `vllm/model_executor/models/llama.py:442-452`

```python
hidden_states = self.embed_input_ids(input_ids)
# Apply conditional zeroing based on fuzz_strength
# Cast to correct dtype to match hidden_states
fuzz_strength = fuzz_strength_per_token.to(dtype=hidden_states.dtype)
# Use mathematical operations to avoid constant folding
# multiplier = 1.0 where fuzz_strength == 0, 0.0 where fuzz_strength >= 1
multiplier = 1.0 - torch.clamp(fuzz_strength, 0.0, 1.0)
hidden_states = hidden_states * multiplier.unsqueeze(-1)
```

**Why**:
- No `if` checks - always executes the same operations for consistent graph
- `clamp(fuzz, 0, 1)` returns 0 when fuzz=0, 1 when fuzz≥1
- `multiplier = 1 - clamp` gives us: fuzz=0 → multiplier=1 (no change), fuzz≥1 → multiplier=0 (zeroed)
- Pure math operations can't be constant-folded by the compiler

---

### How It Works

1. **Initialization**: Create persistent buffer with fixed GPU address
2. **Per-batch**:
   - Fill CPU buffer with expanded per-token fuzz values
   - Copy CPU→GPU at the same fixed address
   - Pass GPU slice to model
3. **CUDA Graph**: Captures the fixed GPU address, updates work correctly
4. **Model**: Applies scaling based on fuzz_strength value at that address

---

### Key Insights

**Why previous attempts failed**:
- ❌ Creating new tensors each iteration: CUDA graphs captured old address
- ❌ Conditional `if fuzz is not None`: Graph saw different code paths
- ❌ Boolean masks `(fuzz > 0)`: Could be constant-folded during compilation

**Why this works**:
- ✅ Persistent buffer: Same address every time
- ✅ In-place updates: Values change, address stays same
- ✅ No conditionals: Always same operations
- ✅ Pure math: Can't be optimized away

---

### Current Behavior

**Expected behavior**:
- `fuzz_strength = 0.0`: multiplier = 1.0 → embeddings unchanged
- `0 < fuzz_strength < 1.0`: partial scaling (multiplier between 0 and 1)
- `fuzz_strength ≥ 1.0`: multiplier = 0.0 → embeddings zeroed (currently used for testing)

## Next Steps

### Option 1: Check Model Loading
Verify the model is actually loading our modified code:
- Check if vLLM is using cached model files
- Verify no compilation cache persists
- Try clearing all caches

### Option 2: Different Model Architecture
Try with a different model that definitely uses this code path:
- Verify Llama 3.2 actually uses the LlamaModel class
- Check for model-specific overrides

### Option 3: Use Different Injection Point
Apply fuzzing in a non-compiled part of the pipeline:
- Apply noise in `embed_input_ids()` before it's called
- Or in the ModelRunner before passing to model

## Debugging Commands

```bash
# Check what parameters model.forward receives
# (torch.compile strips unknown kwargs)

# Disable compilation via environment
export VLLM_TORCH_COMPILE_LEVEL=0

# Or at server startup
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --disable-custom-all-reduce \
  --port 8000
```

## Test Script

`test_fuzzing_comprehensive.py`:
- Uses chat completions API: `/v1/chat/completions`
- Passes `vllm_xargs: {"fuzz_strength": float}` in request
- Tests with 23 diverse instruct-style prompts
- 11 logarithmic fuzz levels (0.0, 1.0→10.0)
- Quick mode: 5 prompts × 7 levels
