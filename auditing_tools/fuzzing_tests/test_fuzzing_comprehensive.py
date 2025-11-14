#!/usr/bin/env python3
"""Comprehensive fuzzing test using chat completions API."""

import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import argparse

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Diverse instruct prompts
PROMPTS = [
    # Factual questions
    "What is the capital city of Japan?",
    "Explain what photosynthesis is.",
    "Who wrote the play Romeo and Juliet?",
    "What is the chemical formula for water?",

    # Math and reasoning
    "Solve this equation: 3x + 7 = 22",
    "What is 15% of 240?",
    "If a train travels 120 km in 2 hours, what is its average speed?",

    # Creative tasks
    "Write a haiku about the ocean.",
    "Describe a sunset in three sentences.",
    "Create a short dialogue between two robots meeting for the first time.",

    # Instructions and how-to
    "Explain how to tie a shoelace in simple steps.",
    "What are three tips for staying healthy?",
    "Describe how a microwave oven works.",

    # Analysis and comparison
    "What are the main differences between cats and dogs as pets?",
    "Compare renewable and non-renewable energy sources.",
    "Explain the difference between weather and climate.",

    # Technical/coding
    "Write a Python function to check if a number is even.",
    "Explain what a for loop does in programming.",
    "What is the difference between a list and a tuple in Python?",

    # Advice and opinion
    "What advice would you give to someone learning a new language?",
    "Why is reading books important?",
    "What are some benefits of regular exercise?",
]

# Default: 5 logarithmic fuzz levels
FUZZ_LEVELS = [0.0] + list(np.logspace(-3, 0, 4))  # 0.0 + 4 levels = 5 total

def clean_text_for_display(text):
    """Clean text by replacing non-printable and problematic unicode characters."""
    # Keep only ASCII printable characters (space through ~)
    # Replace others with '?'
    cleaned = []
    for char in text:
        if 32 <= ord(char) <= 126:  # Printable ASCII range
            cleaned.append(char)
        else:
            cleaned.append('?')
    return ''.join(cleaned)

def send_request(prompt, fuzz_strength, request_id):
    """Send a chat completion request with fuzzing."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.6,
    }

    if fuzz_strength > 0:
        payload["vllm_xargs"] = {"fuzz_strength": fuzz_strength}

    try:
        response = requests.post(BASE_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        completion = result['choices'][0]['message']['content']

        return {
            'id': request_id,
            'prompt': prompt,
            'fuzz': fuzz_strength,
            'completion': completion,
            'success': True,
        }
    except Exception as e:
        return {
            'id': request_id,
            'prompt': prompt,
            'fuzz': fuzz_strength,
            'error': str(e),
            'success': False,
        }

def main():
    parser = argparse.ArgumentParser(description='Fuzzing test for vLLM')
    parser.add_argument('--quick', action='store_true', help='Quick test (5 prompts, 3 levels, 2 samples)')
    parser.add_argument('--max-fuzz', type=float, default=1.0, help='Max fuzz level (default: 10.0)')
    parser.add_argument('--workers', type=int, default=20, help='Concurrent workers (default: 20)')
    parser.add_argument('--samples', type=int, default=5, help='Samples per (prompt, fuzz) config (default: 5)')
    parser.add_argument('--num-levels', type=int, default=5, help='Number of fuzz levels (default: 5)')
    args = parser.parse_args()

    # Use subset for quick mode
    prompts = PROMPTS[:5] if args.quick else PROMPTS

    # Number of samples per config
    num_samples = 2 if args.quick else args.samples

    # Adjust fuzz levels
    if args.quick:
        # Quick mode: just 3 levels (0.0, low, high)
        fuzz_levels = [0.0, 0.001, 0.1]
    else:
        # Generate levels from 0.0 to max_fuzz
        if args.max_fuzz > 1.0:
            num_nonzero_levels = args.num_levels - 1  # -1 for the 0.0 level
            fuzz_levels = [0.0] + list(np.logspace(np.log10(0.00001), np.log10(args.max_fuzz), num_nonzero_levels))
            fuzz_levels = [round(f, 5) for f in fuzz_levels]
        else:
            fuzz_levels = FUZZ_LEVELS[:args.num_levels]

    print("=" * 80)
    print("FUZZING TEST")
    print("=" * 80)
    print(f"Prompts: {len(prompts)}")
    print(f"Fuzz levels: {fuzz_levels}")
    print(f"Samples per config: {num_samples}")
    print(f"Total requests: {len(prompts) * len(fuzz_levels) * num_samples}")
    print("=" * 80)
    print()

    # Generate all requests (with multiple samples per config)
    requests_to_send = []
    req_id = 0
    for prompt in prompts:
        for fuzz in fuzz_levels:
            for sample_idx in range(num_samples):
                requests_to_send.append((prompt, fuzz, f"req_{req_id:04d}_s{sample_idx}"))
                req_id += 1

    print(f"Sending {len(requests_to_send)} requests concurrently...\n")

    # Send requests
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(send_request, p, f, r) for p, f, r in requests_to_send]

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"Progress: {completed}/{len(futures)}", end='\r')
        print()

    # Organize by prompt and fuzz level
    results_by_prompt = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r['success']:
            results_by_prompt[r['prompt']][r['fuzz']].append(r)

    # Display results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    for prompt in prompts:
        if prompt not in results_by_prompt:
            continue

        prompt_results = results_by_prompt[prompt]
        if not prompt_results:
            continue

        print(f"\n{'─' * 80}")
        print(f"PROMPT: {prompt}")
        print(f"{'─' * 80}")

        # Sort fuzz levels
        sorted_fuzz_levels = sorted(prompt_results.keys())

        for fuzz in sorted_fuzz_levels:
            samples = prompt_results[fuzz]
            fuzz_str = f"fuzz={fuzz:>7.5f}"

            # Print all samples for this fuzz level
            for idx, r in enumerate(samples):
                # Clean and truncate completion text (show ~5 lines, ~500 chars)
                completion = clean_text_for_display(r['completion'][:500])
                sample_marker = f"[{idx+1}]" if len(samples) > 1 else "   "

                # Split into lines and display with proper indentation
                lines = completion.split('\n')
                for line_idx, line in enumerate(lines[:5]):  # Show max 5 lines
                    if line_idx == 0:
                        print(f"{fuzz_str} {sample_marker} | {line}")
                    else:
                        # Indent continuation lines
                        print(f"{'':15} | {line}")

    # Summary
    successful = sum(1 for r in results if r['success'])
    print()
    print("=" * 80)
    print(f"SUMMARY: {successful}/{len(results)} successful")
    print("=" * 80)

if __name__ == "__main__":
    main()
