"""
Batch code leakage check for modified prompts.

This script checks if modified prompts contain raw CadQuery code from the original code.
Uses batch inference for efficiency.

Usage:
    python check_leakage.py --input_dir ./sft/filtered_data --output_dir ./sft/leakage_check_results
    python check_leakage.py --input_dir ./sft/filtered_data --model gpt-4.1-2025-04-14 --batch_size 32
"""

import os
import json
import glob
import random
import argparse
import time
import re
from datetime import datetime
from typing import List, Dict, Any

from src.inference import LLM
from config.code_leakage_check import (
    CODE_LEAKAGE_SYSTEM_PROMPT,
    CODE_LEAKAGE_USER_PROMPT_TEMPLATE
)


DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_OUTPUT_DIR = "./results/leakage_check_results"
DEFAULT_BATCH_SIZE = 32
SAVE_EVERY = 32


def load_samples_from_folder(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load samples from batch_label JSON files in the input directory.
    
    Each sample should have:
    - uid
    - modified_prompt
    - ground_truth_code (or generated_code_from_modified_prompt for reference)
    """
    samples = []
    batch_files = glob.glob(os.path.join(input_dir, "batch_label_*.json"))
    
    print(f"Found {len(batch_files)} batch files in {input_dir}")
    
    for batch_file in sorted(batch_files):
        try:
            with open(batch_file, 'r') as f:
                data = json.load(f)
            
            for result in data.get('results', []):
                if result.get('skipped', False):
                    continue
                
                # Extract required fields
                uid = result.get('uid')
                modified_prompt = result.get('modified_prompt', '')
                ground_truth_code = result.get('ground_truth_code', '')
                
                if uid and modified_prompt and ground_truth_code:
                    samples.append({
                        'uid': uid,
                        'modified_prompt': modified_prompt,
                        'original_code': ground_truth_code,
                        'source_file': os.path.basename(batch_file)
                    })
        except Exception as e:
            print(f"Error loading {batch_file}: {e}")
    
    return samples


def load_already_processed(output_dir: str) -> set:
    """Load UIDs of already processed samples from output directory."""
    processed_uids = set()
    
    if not os.path.exists(output_dir):
        return processed_uids
    
    result_files = glob.glob(os.path.join(output_dir, "leakage_check_*.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            for result in data.get('results', []):
                if 'uid' in result:
                    processed_uids.add(result['uid'])
        except Exception as e:
            print(f"Warning: Could not read {result_file}: {e}")
    
    return processed_uids


def build_messages(original_code: str, modified_prompt: str) -> List[Dict[str, str]]:
    """Build chat messages for code leakage check."""
    user_content = CODE_LEAKAGE_USER_PROMPT_TEMPLATE.format(
        original_code=original_code,
        modified_prompt=modified_prompt
    )
    
    return [
        {"role": "system", "content": CODE_LEAKAGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def parse_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from the model."""
    try:
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        
        return {"error": "No JSON found in response", "raw_response": response}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_response": response}


def get_next_batch_number(output_dir: str) -> int:
    """Get the next batch number for saving results."""
    max_num = 0
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, "leakage_check_*.json")):
            match = re.search(r'leakage_check_(\d+)\.json', os.path.basename(f))
            if match:
                max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def save_batch_results(results: List[Dict], batch_num: int, output_dir: str, metadata: Dict) -> str:
    """Save a batch of results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "metadata": {
            **metadata,
            "batch_number": batch_num,
            "batch_size": len(results),
            "saved_at": datetime.now().isoformat()
        },
        "results": results
    }
    
    output_path = os.path.join(output_dir, f"leakage_check_{batch_num:04d}.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved {len(results)} results to {output_path}")
    return output_path


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze batch results and compute statistics."""
    total = len(results)
    contains_code_count = 0
    no_code_count = 0
    error_count = 0
    confidence_scores = []
    
    for r in results:
        parsed = r.get('parsed_response', {})
        if 'error' in parsed:
            error_count += 1
        elif parsed.get('contains_code', False):
            contains_code_count += 1
            if 'confidence' in parsed:
                confidence_scores.append(parsed['confidence'])
        else:
            no_code_count += 1
            if 'confidence' in parsed:
                confidence_scores.append(parsed['confidence'])
    
    stats = {
        'total': total,
        'contains_code': contains_code_count,
        'no_code': no_code_count,
        'errors': error_count,
        'contains_code_pct': 100 * contains_code_count / total if total > 0 else 0,
        'no_code_pct': 100 * no_code_count / total if total > 0 else 0,
    }
    
    if confidence_scores:
        import numpy as np
        stats['avg_confidence'] = np.mean(confidence_scores)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Batch code leakage check for modified prompts')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing batch_label JSON files')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Model to use for inference')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for inference')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY,
                        help='Save results every N samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BATCH CODE LEAKAGE CHECK")
    print("=" * 60)
    print(f"Input Dir:   {args.input_dir}")
    print(f"Output Dir:  {args.output_dir}")
    print(f"Model:       {args.model}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"N Samples:   {args.n_samples or 'all'}")
    print(f"Save Every:  {args.save_every}")
    print(f"Seed:        {args.seed}")
    print("=" * 60)
    
    # Load samples
    print("\nLoading samples...")
    all_samples = load_samples_from_folder(args.input_dir)
    print(f"Loaded {len(all_samples)} samples")
    
    # Random shuffle with fixed seed to get consistent sample selection
    random.seed(args.seed)
    random.shuffle(all_samples)
    
    # Select target n_samples (these are always the same with the same seed)
    if args.n_samples:
        target_samples = all_samples[:args.n_samples]
        print(f"Target samples (with seed={args.seed}): {len(target_samples)}")
    else:
        target_samples = all_samples
        print(f"Processing all {len(target_samples)} samples")
    
    # Load already processed (for resume support)
    print("\nChecking for already processed samples...")
    processed_uids = load_already_processed(args.output_dir)
    print(f"Found {len(processed_uids)} already processed samples")
    
    # Filter out already processed from target samples
    samples = [s for s in target_samples if s['uid'] not in processed_uids]
    print(f"Remaining to process: {len(samples)}")
    
    if len(samples) == 0:
        print("All samples already processed!")
        return
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    llm = LLM(args.model)
    print(f"Using {'API' if llm.use_api else 'transformer'} inference")
    
    # Metadata
    metadata = {
        'model': args.model,
        'input_dir': args.input_dir
    }
    
    # Get starting batch number
    batch_num = get_next_batch_number(args.output_dir)
    
    # Process in chunks
    overall_start = time.time()
    all_results = []
    
    for chunk_start in range(0, len(samples), args.save_every):
        chunk_end = min(chunk_start + args.save_every, len(samples))
        chunk_samples = samples[chunk_start:chunk_end]
        
        print(f"\n[Batch {batch_num}] Processing samples {chunk_start+1}-{chunk_end}/{len(samples)}")
        batch_start = time.time()
        
        # Build messages for batch inference
        messages_list = [
            build_messages(s['original_code'], s['modified_prompt'])
            for s in chunk_samples
        ]
        
        # Batch inference
        print(f"  Running batch inference ({len(messages_list)} samples)...")
        responses = llm.batch_inference(
            messages_list,
            batch_size=args.batch_size
        )
        
        # Process responses
        chunk_results = []
        for sample, response in zip(chunk_samples, responses):
            parsed = parse_response(response)
            
            result = {
                'uid': sample['uid'],
                'source_file': sample['source_file'],
                'modified_prompt': sample['modified_prompt'][:500] + '...' if len(sample['modified_prompt']) > 500 else sample['modified_prompt'],
                'raw_response': response,
                'parsed_response': parsed,
                'contains_code': parsed.get('contains_code', None),
                'confidence': parsed.get('confidence', None)
            }
            chunk_results.append(result)
            
            # Print status
            status = "CODE DETECTED" if parsed.get('contains_code') else "CLEAN"
            if 'error' in parsed:
                status = "PARSE ERROR"
            print(f"  {sample['uid']}: {status}")
        
        batch_time = time.time() - batch_start
        print(f"  Batch time: {batch_time:.1f}s ({batch_time/len(chunk_samples):.2f}s per sample)")
        
        # Save batch results
        save_batch_results(chunk_results, batch_num, args.output_dir, metadata)
        batch_num += 1
        
        all_results.extend(chunk_results)
    
    overall_time = time.time() - overall_start
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    stats = analyze_results(all_results)
    print(f"Total processed: {stats['total']}")
    print(f"Contains code:   {stats['contains_code']} ({stats['contains_code_pct']:.1f}%)")
    print(f"No code:         {stats['no_code']} ({stats['no_code_pct']:.1f}%)")
    print(f"Parse errors:    {stats['errors']}")
    if 'avg_confidence' in stats:
        print(f"Avg confidence:  {stats['avg_confidence']:.1f}")
    
    print(f"\nTotal time: {overall_time:.1f}s ({overall_time/len(all_results):.2f}s per sample)")
    print(f"Results saved to: {args.output_dir}")
    
    # Cleanup
    llm.unload()


if __name__ == "__main__":
    main()
