#!/usr/bin/env python3
"""
Processing utilities for filtering and converting misleading samples.

This module provides functions for:
- Filtering samples based on chamfer distance criteria
- Converting filtered samples to SFT format
"""

import json
import glob
import os
import argparse
import math
from typing import Dict, List, Tuple, Optional


def filter_samples(
    eval_dir: str,
    misleading_dir: str,
    output_file: Optional[str] = None,
    ratio_threshold: float = 10,
    misleading_cd_min: float = 0.0002,
    original_cd_max: float = 0.0002
) -> Dict:
    """
    Filter samples based on chamfer distance criteria.
    
    Args:
        eval_dir: Directory containing evaluation results (batch_*.json files)
        misleading_dir: Directory containing misleading samples (misleading_batch_*.json files)
        output_file: Path to output JSON file for filtered samples. If None, defaults to
                     {misleading_dir}/selected_misleading_samples_ratio10.json
        ratio_threshold: Minimum ratio of misleading_cd to original_cd
        misleading_cd_min: Minimum chamfer distance for misleading prompts
        original_cd_max: Maximum chamfer distance for original prompts
    
    Returns:
        Dictionary with filtered samples and metadata
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(misleading_dir, "selected_misleading_samples_ratio10.json")
    # Load evaluation results (misleading prompt evaluations)
    # Store all evaluation results keyed by uid
    eval_results_by_uid = {}
    eval_result_count = 0
    for f in sorted(glob.glob(os.path.join(eval_dir, "batch_*.json"))):
        try:
            with open(f) as fp:
                data = json.load(fp)
                for result in data.get("results", []):
                    uid = result.get("uid")
                    misleading_cd = result.get("chamfer_distance")
                    
                    # Check if misleading_cd is valid (not None and not NaN)
                    if misleading_cd is not None:
                        # Handle NaN values (could come from numpy or math operations)
                        if isinstance(misleading_cd, float) and math.isnan(misleading_cd):
                            # Skip NaN values
                            continue
                        
                        # Only include successful evaluations with valid chamfer_distance
                        if result.get("success") and misleading_cd > 0:
                            if uid not in eval_results_by_uid:
                                eval_results_by_uid[uid] = []
                            eval_results_by_uid[uid].append(misleading_cd)
                            eval_result_count += 1
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    print(f"Loaded {eval_result_count} evaluation results for {len(eval_results_by_uid)} unique UIDs")
    
    # Load misleading prompts with original_cd
    misleading_samples = {}
    for f in sorted(glob.glob(os.path.join(misleading_dir, "misleading_batch_*.json"))):
        try:
            with open(f) as fp:
                data = json.load(fp)
                for result in data.get("results", []):
                    if result.get("success"):
                        uid = result.get("uid")
                        config_name = result.get("config_name")
                        original_cd = result.get("original_cd")
                        
                        if original_cd is None:
                            continue
                        
                        key = (uid, config_name)
                        misleading_samples[key] = result
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    print(f"Loaded {len(misleading_samples)} misleading samples")
    
    # Match evaluation results with misleading samples
    # For each (uid, config_name) combination, we need to match with an evaluation result
    # Since evaluation results don't have config_name, we'll match by uid
    # For each uid, there might be multiple evaluation results (one per k value)
    # We'll match them in order: first evaluation result with first config_name (k1), etc.
    
    # Group misleading samples by uid
    samples_by_uid = {}
    for (uid, config_name), sample in misleading_samples.items():
        if uid not in samples_by_uid:
            samples_by_uid[uid] = []
        samples_by_uid[uid].append((config_name, sample))
    
    # Sort by config_name to ensure consistent matching (k1, k2, k3, etc.)
    for uid in samples_by_uid:
        samples_by_uid[uid].sort(key=lambda x: x[0])
    
    filtered = []
    successful_samples = 0
    failed_samples = 0
    
    for uid, config_samples in samples_by_uid.items():
        # If uid has no successful evaluation results (all evaluations failed or invalid),
        # count all samples for this uid as failed. This handles the case where
        # original_cd is valid but misleading_cd evaluation failed (success=False, None, or NaN).
        if uid not in eval_results_by_uid:
            failed_samples += len(config_samples)
            continue
        
        eval_cds = eval_results_by_uid[uid].copy()  # Copy to track used results
        used_indices = set()  # Track which evaluation results have been used
        
        # Match in order: first config_name with first evaluation result, etc.
        for idx, (config_name, sample) in enumerate(config_samples):
            original_cd = sample.get("original_cd")
            
            # Check if original_cd is valid (not None, not NaN, and > 0)
            if original_cd is None:
                failed_samples += 1
                continue
            
            # Handle NaN values for original_cd
            if isinstance(original_cd, float) and math.isnan(original_cd):
                failed_samples += 1
                continue
            
            if original_cd <= 0:
                failed_samples += 1
                continue
            
            matched = False
            
            # First try to match with evaluation result at the same index (if not used)
            if idx < len(eval_cds) and idx not in used_indices:
                misleading_cd = eval_cds[idx]
                
                # Check if misleading_cd is valid (not None, not NaN, and > 0)
                if (misleading_cd is not None and 
                    not (isinstance(misleading_cd, float) and math.isnan(misleading_cd)) and
                    misleading_cd > 0 and original_cd > 0):
                    
                    ratio = misleading_cd / original_cd
                    
                    # Check if ratio is valid (not NaN)
                    if not (isinstance(ratio, float) and math.isnan(ratio)):
                        if (ratio > ratio_threshold and 
                            misleading_cd > misleading_cd_min and 
                            original_cd < original_cd_max):
                            
                            filtered_sample = sample.copy()
                            filtered_sample["misleading_cd"] = misleading_cd
                            filtered_sample["ratio"] = ratio
                            filtered_sample["misleading_success"] = True
                            filtered.append(filtered_sample)
                            successful_samples += 1
                            used_indices.add(idx)
                            matched = True
            
            # If no match found, try all remaining evaluation results for this uid
            if not matched:
                for eval_idx, misleading_cd in enumerate(eval_cds):
                    if eval_idx in used_indices:
                        continue
                    
                    # Check if misleading_cd is valid (not None, not NaN, and > 0)
                    if (misleading_cd is not None and 
                        not (isinstance(misleading_cd, float) and math.isnan(misleading_cd)) and
                        misleading_cd > 0 and original_cd > 0):
                        
                        ratio = misleading_cd / original_cd
                        
                        # Check if ratio is valid (not NaN)
                        if not (isinstance(ratio, float) and math.isnan(ratio)):
                            if (ratio > ratio_threshold and 
                                misleading_cd > misleading_cd_min and 
                                original_cd < original_cd_max):
                                
                                filtered_sample = sample.copy()
                                filtered_sample["misleading_cd"] = misleading_cd
                                filtered_sample["ratio"] = ratio
                                filtered_sample["misleading_success"] = True
                                filtered.append(filtered_sample)
                                successful_samples += 1
                                used_indices.add(eval_idx)
                                matched = True
                                break
            
            # If no match found, count as failed
            # This handles the case where:
            # - original_cd is valid but no valid misleading_cd evaluation exists for this sample
            # - misleading_cd is None, NaN, or evaluation failed (success=False)
            # - All available misleading_cd values don't meet the filter criteria
            if not matched:
                failed_samples += 1
    
    # Save filtered results
    output_data = {
        "metadata": {
            "total_samples": len(filtered),
            "ratio_threshold": ratio_threshold,
            "misleading_cd_min": misleading_cd_min,
            "original_cd_max": original_cd_max,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples
        },
        "samples": filtered
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as fp:
        json.dump(output_data, fp, indent=2, ensure_ascii=False)
    
    print(f"Filtered {len(filtered)} samples from {len(misleading_samples)} misleading samples")
    print(f"  Successful: {successful_samples}")
    print(f"  Failed: {failed_samples}")
    print(f"Saved to {output_file}")
    
    return output_data


def main():
    """Command-line interface for filtering samples."""
    parser = argparse.ArgumentParser(
        description='Filter misleading samples based on chamfer distance criteria'
    )
    parser.add_argument(
        '--eval_dir',
        type=str,
        required=True,
        help='Directory containing evaluation results (batch_*.json files)'
    )
    parser.add_argument(
        '--misleading_dir',
        type=str,
        required=True,
        help='Directory containing misleading samples (misleading_batch_*.json files)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to output JSON file for filtered samples. If not provided, defaults to '
             '{misleading_dir}/selected_misleading_samples_ratio10.json'
    )
    parser.add_argument(
        '--ratio_threshold',
        type=float,
        default=10,
        help='Minimum ratio of misleading_cd to original_cd (default: 10)'
    )
    parser.add_argument(
        '--misleading_cd_min',
        type=float,
        default=0.0002,
        help='Minimum chamfer distance for misleading prompts (default: 0.0002)'
    )
    parser.add_argument(
        '--original_cd_max',
        type=float,
        default=0.0002,
        help='Maximum chamfer distance for original prompts (default: 0.0002)'
    )
    
    args = parser.parse_args()
    
    filter_samples(
        eval_dir=args.eval_dir,
        misleading_dir=args.misleading_dir,
        output_file=args.output_file,
        ratio_threshold=args.ratio_threshold,
        misleading_cd_min=args.misleading_cd_min,
        original_cd_max=args.original_cd_max
    )


if __name__ == "__main__":
    main()
