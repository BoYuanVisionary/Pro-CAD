"""
Post-processing script for clarification pipeline results.

This script evaluates clarification pipeline results by:
1. Calculating success rate (percentage with valid CADQuery code)
2. Calculating mean and median Chamfer Distance
3. Evaluating question quality using judge LLM (generated vs ground truth questions)
4. Evaluating ambiguity resolution using judge LLM (how well corrected prompt resolves ambiguities)
"""

import json
import os
import glob
import argparse
import re
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.inference import LLM
from src.evaluation import calculate_question_score
from config.clarification import (
    JUDGE_QUESTION_QUALITY_SYSTEM_PROMPT,
    JUDGE_QUESTION_QUALITY_USER_TEMPLATE,
    JUDGE_AMBIGUITY_RESOLUTION_SYSTEM_PROMPT,
    JUDGE_AMBIGUITY_RESOLUTION_USER_TEMPLATE
)


DEFAULT_JUDGE_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_BATCH_SIZE = 8


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from LLM response that might be wrapped in markdown code blocks."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON directly
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def load_clarification_results(results_dir: str, single_file: Optional[str] = None, source_json_path: Optional[str] = None) -> List[Dict]:
    """Load all clarification result files from directory or a single file.
    
    Args:
        results_dir: Directory containing clarification result files
        single_file: Optional single file to load
        source_json_path: Optional path to source JSON (selected_misleading_samples_ratio10.json) to load missing fields
    """
    results = []
    
    # Load source JSON if provided (to get what_changed and ambiguity_scan)
    source_data = {}
    if source_json_path and os.path.exists(source_json_path):
        try:
            with open(source_json_path, 'r') as f:
                source_json = json.load(f)
                source_samples = source_json.get("samples", [])
                # Create a lookup by (uid, k)
                for sample in source_samples:
                    uid = sample.get("uid")
                    k = sample.get("k", "unknown")
                    key = (uid, k)
                    source_data[key] = sample
            print(f"Loaded source JSON with {len(source_data)} samples for missing field lookup")
        except Exception as e:
            print(f"Warning: Could not load source JSON {source_json_path}: {e}")
    
    if single_file:
        # Load single file
        if os.path.exists(single_file):
            try:
                with open(single_file, 'r') as fp:
                    result = json.load(fp)
                    # Fill in missing fields from source JSON
                    if source_data:
                        uid = result.get("uid")
                        k = result.get("k", "unknown")
                        key = (uid, k)
                        if key in source_data:
                            source_sample = source_data[key]
                            if not result.get("what_changed"):
                                result["what_changed"] = source_sample.get("what_changed", "")
                            if not result.get("ambiguity_scan"):
                                result["ambiguity_scan"] = source_sample.get("ambiguity_scan", "")
                    results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {single_file}: {e}")
        else:
            print(f"Warning: File not found: {single_file}")
        return results
    
    # Load individual result files
    seen_keys = set()  # (uid, k) to avoid duplicates when both file types exist
    for f in glob.glob(os.path.join(results_dir, "clarification_result_*.json")):
        try:
            with open(f, 'r') as fp:
                result = json.load(fp)
                uid = result.get("uid")
                k = result.get("k", "unknown")
                key = (uid, k)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Fill in missing fields from source JSON
                if source_data and key in source_data:
                    source_sample = source_data[key]
                    if not result.get("what_changed"):
                        result["what_changed"] = source_sample.get("what_changed", "")
                    if not result.get("ambiguity_scan"):
                        result["ambiguity_scan"] = source_sample.get("ambiguity_scan", "")
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    # Also try loading from all_results_*.json files (skip if already loaded from clarification_result_*)
    for f in glob.glob(os.path.join(results_dir, "all_results_*.json")):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if "results" in data:
                    for result in data["results"]:
                        uid = result.get("uid")
                        k = result.get("k", "unknown")
                        key = (uid, k)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        # Fill in missing fields from source JSON
                        if source_data and key in source_data:
                            source_sample = source_data[key]
                            if not result.get("what_changed"):
                                result["what_changed"] = source_sample.get("what_changed", "")
                            if not result.get("ambiguity_scan"):
                                result["ambiguity_scan"] = source_sample.get("ambiguity_scan", "")
                        results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return results


def evaluate_question_quality(
    judge_llm: LLM,
    misleading_prompt: str,
    ground_truth_questions: str,
    generated_questions: List[str],
    detailed_logging: bool = False
) -> Dict:
    """Evaluate quality of generated questions using judge LLM."""
    # Parse and clean ground truth questions (remove ANSWER_TO_QUESTIONS section)
    gt_questions_list = parse_ground_truth_questions(ground_truth_questions)
    # Format for judge prompt
    if gt_questions_list:
        gt_questions_str = "\n".join(f"- {q}" for q in gt_questions_list)
    else:
        # Fallback: use raw string but clean it
        gt_questions_str = ground_truth_questions
        # Remove ANSWER_TO_QUESTIONS section
        for pattern in [r'\n\n\d+\)\s*ANSWER_TO_QUESTIONS.*$', r'\n\nANSWER_TO_QUESTIONS.*$', r'ANSWER_TO_QUESTIONS.*$']:
            gt_questions_str = re.sub(pattern, '', gt_questions_str, flags=re.DOTALL | re.IGNORECASE)
        gt_questions_str = gt_questions_str.strip()
    
    # Format generated questions
    gen_questions_str = "\n".join(f"- {q}" for q in generated_questions)
    
    user_prompt = JUDGE_QUESTION_QUALITY_USER_TEMPLATE.format(
        misleading_prompt=misleading_prompt,
        ground_truth_questions=gt_questions_str,
        generated_questions=gen_questions_str
    )
    
    messages = [
        {"role": "system", "content": JUDGE_QUESTION_QUALITY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    if detailed_logging:
        print("\n" + "="*60)
        print("QUESTION QUALITY JUDGE - LLM REQUEST")
        print("="*60)
        print("\nSystem Prompt:")
        print(JUDGE_QUESTION_QUALITY_SYSTEM_PROMPT)
        print("\nUser Prompt:")
        print(user_prompt)
        print("="*60)
    
    try:
        response = judge_llm.inference(messages)
        
        if detailed_logging:
            print("\nQUESTION QUALITY JUDGE - LLM RESPONSE")
            print("="*60)
            print(response)
            print("="*60)
        
        evaluation = extract_json_from_response(response)
        
        if evaluation is None:
            return {
                "success": False,
                "error": "Could not parse JSON from judge response",
                "raw_response": response
            }
        
        evaluation["success"] = True
        return evaluation
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def evaluate_ambiguity_resolution(
    judge_llm: LLM,
    original_prompt: str,
    misleading_prompt: str,
    clarified_prompt: str,
    what_changed: str = "",
    ambiguity_scan: str = "",
    detailed_logging: bool = False
) -> Dict:
    """Evaluate how well clarified prompt resolves ambiguities using judge LLM."""
    # Use default values if not provided
    if not what_changed:
        what_changed = "Not available"
    if not ambiguity_scan:
        ambiguity_scan = "Not available"
    
    user_prompt = JUDGE_AMBIGUITY_RESOLUTION_USER_TEMPLATE.format(
        original_prompt=original_prompt,
        misleading_prompt=misleading_prompt,
        what_changed=what_changed,
        ambiguity_scan=ambiguity_scan,
        clarified_prompt=clarified_prompt
    )
    
    messages = [
        {"role": "system", "content": JUDGE_AMBIGUITY_RESOLUTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    if detailed_logging:
        print("\n" + "="*60)
        print("AMBIGUITY RESOLUTION JUDGE - LLM REQUEST")
        print("="*60)
        print("\nSystem Prompt:")
        print(JUDGE_AMBIGUITY_RESOLUTION_SYSTEM_PROMPT)
        print("\nUser Prompt:")
        print(user_prompt)
        print("="*60)
    
    try:
        response = judge_llm.inference(messages)
        
        if detailed_logging:
            print("\nAMBIGUITY RESOLUTION JUDGE - LLM RESPONSE")
            print("="*60)
            print(response)
            print("="*60)
        
        evaluation = extract_json_from_response(response)
        
        if evaluation is None:
            return {
                "success": False,
                "error": "Could not parse JSON from judge response",
                "raw_response": response
            }
        
        evaluation["success"] = True
        return evaluation
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def extract_corrected_prompt(result: Dict) -> Optional[str]:
    """Extract the corrected/standardized prompt from result."""
    # Try step3_corrected_description
    step3 = result.get("step3_corrected_description", "")
    if step3:
        # Extract from JSON if needed
        json_match = re.search(r'"standardized_prompt"\s*:\s*"([^"]+)"', step3)
        if json_match:
            return json_match.group(1).replace('\\n', '\n')
    
    # Try final_prompt or corrected_prompt fields
    for key in ["final_prompt", "corrected_prompt", "standardized_prompt"]:
        if key in result:
            return result[key]
    
    return None


def is_failed_case(result: Dict) -> bool:
    """Check if this is a failed case where the model didn't detect ambiguity.
    
    Failed cases:
    - step1_analysis.is_misleading is False (model didn't detect ambiguity)
    - step3_corrected_description contains is_misleading: true (model still thinks it's misleading after clarification)
    """
    step1 = result.get("step1_analysis", {})
    if step1:
        is_misleading = step1.get("is_misleading")
        if is_misleading is False:
            return True
    
    step3 = result.get("step3_corrected_description", "")
    if step3:
        # Check if step3 response says is_misleading: true
        if re.search(r'"is_misleading"\s*:\s*true', step3, re.IGNORECASE):
            return True
    
    return False


def extract_generated_questions(result: Dict) -> List[str]:
    """Extract generated questions from result."""
    step1 = result.get("step1_analysis", {})
    questions = step1.get("questions", [])
    if isinstance(questions, list):
        return questions
    elif isinstance(questions, str):
        # Try to parse from string
        questions_list = []
        for line in questions.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+[\.\)]', line)):
                questions_list.append(re.sub(r'^[-•]\s*|\d+[\.\)]\s*', '', line))
        return questions_list if questions_list else [questions]
    return []


def parse_ground_truth_questions(ground_truth_questions) -> List[str]:
    """Parse ground truth questions into a list.
    
    Removes the "ANSWER_TO_QUESTIONS" section and everything after it.
    """
    if isinstance(ground_truth_questions, list):
        return ground_truth_questions
    elif isinstance(ground_truth_questions, str):
        # Remove "ANSWER_TO_QUESTIONS" section and everything after it
        # Look for patterns like "\n\n5) ANSWER_TO_QUESTIONS\n" or "ANSWER_TO_QUESTIONS"
        answer_patterns = [
            r'\n\n\d+\)\s*ANSWER_TO_QUESTIONS.*$',
            r'\n\nANSWER_TO_QUESTIONS.*$',
            r'ANSWER_TO_QUESTIONS.*$'
        ]
        
        cleaned_text = ground_truth_questions
        for pattern in answer_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Try to parse from string (may have bullet points or numbers)
        questions_list = []
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+[\.\)]', line)):
                # Remove bullet points and numbers
                clean_line = re.sub(r'^[-•]\s*|\d+[\.\)]\s*', '', line)
                if clean_line:
                    questions_list.append(clean_line)
        return questions_list if questions_list else [cleaned_text.strip()] if cleaned_text.strip() else []
    return []


def _save_batch_results(processed_results, results_dir, judge_model, skip_judge_evaluation, 
                       stats, output_dir, batch_num=0, batch_indices=None):
    """Save results after processing a batch (both scores included)."""
    import numpy as np
    
    # Calculate current statistics
    success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
    cd_mean = np.mean(stats["chamfer_distances"]) if stats["chamfer_distances"] else None
    cd_median = np.median(stats["chamfer_distances"]) if stats["chamfer_distances"] else None
    
    # Get only the results for this batch
    if batch_indices:
        batch_results = [processed_results[i] for i in batch_indices]
    else:
        batch_results = processed_results
    
    # Create summary
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "results_dir": results_dir,
            "judge_model": judge_model if not skip_judge_evaluation else None,
            "batch_num": batch_num,
            "batch_size": len(batch_results)
        },
        "statistics": {
            "success_rate": {
                "percentage": success_rate,
                "successful": stats["success"],
                "failed": stats["failed"],
                "total": stats["total"]
            },
            "chamfer_distance": {
                "mean": float(cd_mean) if cd_mean is not None else None,
                "median": float(cd_median) if cd_median is not None else None,
                "count": len(stats["chamfer_distances"])
            },
            "question_quality_evaluations_count": len(stats["question_quality_evaluations"]),
            "ambiguity_resolution_evaluations_count": len(stats["ambiguity_resolution_evaluations"])
        },
        "results": batch_results
    }
    
    # Save to batch-specific file
    os.makedirs(output_dir, exist_ok=True)
    batch_output_file = os.path.join(output_dir, f'post_process_batch_{batch_num:04d}.json')
    with open(batch_output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return batch_output_file


def main(
    results_dir: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    skip_judge_evaluation: bool = False,
    detailed_logging: bool = False,
    n_samples: Optional[int] = None,
    single_file: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    source_json_path: Optional[str] = None
):
    """Main post-processing function."""
    print("=" * 60)
    print("POST-PROCESSING CLARIFICATION RESULTS")
    print("=" * 60)
    print(f"Results Directory: {results_dir}")
    print(f"Judge Model: {judge_model}")
    print(f"Skip Judge Evaluation: {skip_judge_evaluation}")
    print(f"Detailed Logging: {detailed_logging}")
    print(f"Max Samples: {n_samples}")
    print(f"Single File: {single_file}")
    print(f"Batch Size: {batch_size}")
    print(f"Source JSON: {source_json_path}")
    print("=" * 60)
    
    # Load results
    print("\nLoading clarification results...")
    results = load_clarification_results(results_dir, single_file=single_file, source_json_path=source_json_path)
    print(f"Loaded {len(results)} results")
    
    if len(results) == 0:
        print("No results found. Exiting.")
        return
    
    # Limit to n_samples if specified
    if n_samples and n_samples < len(results):
        results = results[:n_samples]
        print(f"Limited to first {n_samples} samples")
    
    # Load existing batch files to check for resume
    existing_processed = {}
    if not single_file:
        batch_files = sorted(glob.glob(os.path.join(results_dir, "post_process_batch_*.json")))
        if batch_files:
            print(f"Found {len(batch_files)} existing batch files (resume mode)")
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'r') as f:
                        existing_data = json.load(f)
                        existing_results = existing_data.get("results", [])
                        for existing_result in existing_results:
                            uid = existing_result.get("uid")
                            k = existing_result.get("k", "unknown")
                            key = (uid, k)
                            # Store existing result (will skip if both evaluations are done)
                            existing_processed[key] = existing_result
                except Exception as e:
                    print(f"Warning: Could not load batch file {batch_file}: {e}")
            if existing_processed:
                print(f"Loaded {len(existing_processed)} existing results from batch files")
    
    # Initialize judge LLM if needed
    judge_llm = None
    if not skip_judge_evaluation:
        print(f"\nInitializing judge LLM: {judge_model}")
        judge_llm = LLM(model_name=judge_model)
    
    # Process results
    stats = {
        "total": len(results),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "chamfer_distances": [],
        "question_quality_evaluations": [],
        "ambiguity_resolution_evaluations": []
    }
    
    processed_results = []
    
    print(f"\nProcessing {len(results)} results...")
    for i, result in enumerate(results):
        uid = result.get("uid", f"unknown_{i}")
        k = result.get("k", "unknown")
        key = (uid, k)
        
        # Handle k=0 samples (baseline - should not be misleading)
        if k == 0:
            processed_result = result.copy()
            
            # Check if model correctly identified as not misleading
            step1_analysis = result.get("step1_analysis", {})
            is_misleading = step1_analysis.get("is_misleading", True) if step1_analysis else True
            
            if not is_misleading:
                # Model correctly identified it as not misleading - give perfect scores
                print(f"  [{i+1}/{len(results)}] Processing {uid} k={k} (k=0, correctly identified - perfect scores)")
                
                # Add perfect question quality evaluation
                processed_result["question_quality_evaluation"] = {
                    "success": True,
                    "score": 1.0,
                    "reasoning": "k=0 baseline sample correctly identified as not misleading",
                    "raw_response": "Perfect - no questions needed for non-misleading baseline"
                }
                
                # Add perfect ambiguity resolution evaluation  
                processed_result["ambiguity_resolution_evaluation"] = {
                    "success": True,
                    "score": 1.0,
                    "reasoning": "k=0 baseline sample correctly identified as not misleading",
                    "raw_response": "Perfect - no ambiguity resolution needed for non-misleading baseline"
                }
                
                stats["question_quality_evaluations"].append({"score": 1.0})
                stats["ambiguity_resolution_evaluations"].append({"score": 1.0})
            else:
                # Model incorrectly thought it was misleading - skip evaluation
                print(f"  [{i+1}/{len(results)}] Skipping {uid} k={k} (k=0 but incorrectly identified as misleading)")
            
            # Still count success/failure for CD stats
            eval_result = result.get("evaluation")
            if eval_result and eval_result.get("success"):
                stats["success"] += 1
                cd = eval_result.get("chamfer_distance")
                if cd is not None:
                    stats["chamfer_distances"].append(cd)
            else:
                stats["failed"] += 1
            if result.get("skipped"):
                stats["skipped"] += 1
            
            processed_results.append(processed_result)
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(results)} results...")
        
        # Check if already processed (resume support)
        if key in existing_processed and not skip_judge_evaluation:
            existing_result = existing_processed[key]
            has_question_eval = existing_result.get("question_quality_evaluation", {}).get("success", False)
            has_ambiguity_eval = existing_result.get("ambiguity_resolution_evaluation", {}).get("success", False)
            
            # Skip if both evaluations are already done
            if has_question_eval and has_ambiguity_eval:
                print(f"  [{i+1}/{len(results)}] Skipping {uid} k={k} (already fully processed)")
                processed_results.append(existing_result)
                continue
            else:
                # Use existing result as base, only evaluate missing parts
                processed_result = existing_result.copy()
                print(f"  [{i+1}/{len(results)}] Resuming {uid} k={k} (partial evaluation exists)")
        else:
            processed_result = result.copy()
        
        # Check success (has valid CADQuery code) - use original result, not processed_result
        eval_result = result.get("evaluation")
        if eval_result and eval_result.get("success"):
            stats["success"] += 1
            cd = eval_result.get("chamfer_distance")
            if cd is not None:
                stats["chamfer_distances"].append(cd)
        else:
            stats["failed"] += 1
        
        if result.get("skipped"):
            stats["skipped"] += 1
        
        processed_results.append(processed_result)
    
    # Batch judge evaluations if needed
    if not skip_judge_evaluation and judge_llm:
        print(f"\n{'='*60}")
        print("BATCH JUDGE EVALUATION")
        print(f"{'='*60}")
        
        # Collect all evaluation requests
        question_quality_batch = []  # List of (idx, messages, ground_truth_questions, result)
        ambiguity_resolution_batch = []  # List of (idx, messages, result)
        
        for i, (result, processed_result) in enumerate(zip(results, processed_results)):
            uid = result.get("uid", f"unknown_{i}")
            k = result.get("k", "unknown")
            
            # Skip k=0 samples (no questions to evaluate)
            if k == 0:
                continue
            
            # Check what needs to be evaluated
            needs_question_eval = not processed_result.get("question_quality_evaluation", {}).get("success", False)
            needs_ambiguity_eval = not processed_result.get("ambiguity_resolution_evaluation", {}).get("success", False)
            
            # Question quality evaluation
            if needs_question_eval:
                ground_truth_questions = result.get("ground_truth_questions", "")
                generated_questions = extract_generated_questions(result)
                misleading_prompt = result.get("misleading_prompt", "")
                
                if ground_truth_questions and generated_questions and misleading_prompt:
                    # Parse and clean ground truth questions (remove ANSWER_TO_QUESTIONS section)
                    gt_questions_list = parse_ground_truth_questions(ground_truth_questions)
                    # Format for judge prompt
                    if gt_questions_list:
                        gt_questions_str = "\n".join(f"- {q}" for q in gt_questions_list)
                    else:
                        # Fallback: use raw string but clean it
                        gt_questions_str = ground_truth_questions
                        # Remove ANSWER_TO_QUESTIONS section
                        for pattern in [r'\n\n\d+\)\s*ANSWER_TO_QUESTIONS.*$', r'\n\nANSWER_TO_QUESTIONS.*$', r'ANSWER_TO_QUESTIONS.*$']:
                            gt_questions_str = re.sub(pattern, '', gt_questions_str, flags=re.DOTALL | re.IGNORECASE)
                        gt_questions_str = gt_questions_str.strip()
                    
                    # Format generated questions
                    gen_questions_str = "\n".join(f"- {q}" for q in generated_questions)
                    
                    user_prompt = JUDGE_QUESTION_QUALITY_USER_TEMPLATE.format(
                        misleading_prompt=misleading_prompt,
                        ground_truth_questions=gt_questions_str,
                        generated_questions=gen_questions_str
                    )
                    
                    messages = [
                        {"role": "system", "content": JUDGE_QUESTION_QUALITY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    question_quality_batch.append((i, messages, ground_truth_questions, result))
            
            # Ambiguity resolution evaluation
            if needs_ambiguity_eval:
                original_prompt = result.get("original_prompt", "")
                clarified_prompt = extract_corrected_prompt(result)
                misleading_prompt = result.get("misleading_prompt", "")
                
                # Check if this is a failed case (model didn't detect ambiguity)
                if is_failed_case(result):
                    print(f"  Warning: Sample {uid} k={k} is a failed case (model didn't detect ambiguity), assigning score 0.0")
                    step1 = result.get("step1_analysis", {})
                    is_misleading_step1 = step1.get("is_misleading")
                    
                    if is_misleading_step1 is False:
                        reasoning = "Model determined the prompt is not misleading (is_misleading=false in step1). Failed to detect ambiguity."
                        label = "failed_detection_step1"
                    else:
                        reasoning = "Model still determined the prompt is misleading after clarification (is_misleading=true in step3). Failed to resolve ambiguity."
                        label = "failed_detection_step3"
                    
                    processed_results[i]["ambiguity_resolution_evaluation"] = {
                        "success": True,
                        "score": 0.0,
                        "reasoning": reasoning,
                        "label": label
                    }
                    stats["ambiguity_resolution_evaluations"].append({
                        "success": True,
                        "score": 0.0,
                        "reasoning": reasoning,
                        "label": label
                    })
                # If clarified_prompt is missing, assign score 0.0 automatically
                elif not clarified_prompt:
                    print(f"  Warning: Sample {uid} k={k} missing clarified_prompt, assigning score 0.0")
                    processed_results[i]["ambiguity_resolution_evaluation"] = {
                        "success": True,
                        "score": 0.0,
                        "reasoning": "Clarified prompt is missing or could not be extracted from step3_corrected_description. Cannot evaluate ambiguity resolution.",
                        "label": "missing_clarified_prompt"
                    }
                    stats["ambiguity_resolution_evaluations"].append({
                        "success": True,
                        "score": 0.0,
                        "reasoning": "Clarified prompt is missing",
                        "label": "missing_clarified_prompt"
                    })
                elif not original_prompt:
                    print(f"  Warning: Sample {uid} k={k} missing original_prompt, assigning score 0.0")
                    processed_results[i]["ambiguity_resolution_evaluation"] = {
                        "success": True,
                        "score": 0.0,
                        "reasoning": "Original prompt is missing. Cannot evaluate ambiguity resolution.",
                        "label": "missing_original_prompt"
                    }
                    stats["ambiguity_resolution_evaluations"].append({
                        "success": True,
                        "score": 0.0,
                        "reasoning": "Original prompt is missing",
                        "label": "missing_original_prompt"
                    })
                elif not misleading_prompt:
                    print(f"  Warning: Sample {uid} k={k} missing misleading_prompt, assigning score 0.0")
                    processed_results[i]["ambiguity_resolution_evaluation"] = {
                        "success": True,
                        "score": 0.0,
                        "reasoning": "Misleading prompt is missing. Cannot evaluate ambiguity resolution.",
                        "label": "missing_misleading_prompt"
                    }
                    stats["ambiguity_resolution_evaluations"].append({
                        "success": True,
                        "score": 0.0,
                        "reasoning": "Misleading prompt is missing",
                        "label": "missing_misleading_prompt"
                    })
                else:
                    # All required fields present, proceed with normal evaluation
                    # Extract actual ambiguities from result
                    what_changed = result.get("what_changed", "")
                    ambiguity_scan = result.get("ambiguity_scan", "")
                    
                    # If not in result, try to find in any loaded sample data
                    if not what_changed or not ambiguity_scan:
                        step1 = result.get("step1_analysis", {})
                        if not what_changed:
                            what_changed = step1.get("what_changed", "")
                        if not ambiguity_scan:
                            ambiguity_scan = step1.get("ambiguity_scan", "")
                    
                    # Use default values if not provided
                    if not what_changed:
                        what_changed = "Not available"
                    if not ambiguity_scan:
                        ambiguity_scan = "Not available"
                    
                    user_prompt = JUDGE_AMBIGUITY_RESOLUTION_USER_TEMPLATE.format(
                        original_prompt=original_prompt,
                        misleading_prompt=misleading_prompt,
                        what_changed=what_changed,
                        ambiguity_scan=ambiguity_scan,
                        clarified_prompt=clarified_prompt
                    )
                    
                    messages = [
                        {"role": "system", "content": JUDGE_AMBIGUITY_RESOLUTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    ambiguity_resolution_batch.append((i, messages, result))
        
        # Group samples into batches and process both scores together
        # Combine both evaluation types and group by sample indices
        all_evaluation_indices = set()
        for idx, _, _, _ in question_quality_batch:
            all_evaluation_indices.add(idx)
        for idx, _, _ in ambiguity_resolution_batch:
            all_evaluation_indices.add(idx)
        
        all_evaluation_indices = sorted(all_evaluation_indices)
        total_batches = (len(all_evaluation_indices) + batch_size - 1) // batch_size
        
        print(f"\nProcessing {len(all_evaluation_indices)} samples in {total_batches} batches of {batch_size}...")
        
        # Process in batches
        for batch_num in range(1, total_batches + 1):
            batch_start = (batch_num - 1) * batch_size
            batch_end = min(batch_start + batch_size, len(all_evaluation_indices))
            batch_indices = all_evaluation_indices[batch_start:batch_end]
            
            print(f"\n{'='*60}")
            print(f"BATCH {batch_num}/{total_batches} (samples {batch_start+1}-{batch_end})")
            print(f"{'='*60}")
            
            # Get question quality requests for this batch
            batch_question_quality = [(idx, msgs, gt_q, res) for idx, msgs, gt_q, res in question_quality_batch if idx in batch_indices]
            # Get ambiguity resolution requests for this batch
            batch_ambiguity_resolution = [(idx, msgs, res) for idx, msgs, res in ambiguity_resolution_batch if idx in batch_indices]
            
            # Process question quality for this batch
            if batch_question_quality:
                print(f"\n  Processing {len(batch_question_quality)} question quality evaluations...")
                messages_list = [msgs for _, msgs, _, _ in batch_question_quality]
                
                if detailed_logging:
                    print("\n" + "="*60)
                    print(f"QUESTION QUALITY JUDGE - BATCH {batch_num} LLM REQUEST")
                    print("="*60)
                    for j, (idx, msgs, _, _) in enumerate(batch_question_quality[:3]):
                        print(f"\n--- Sample {idx+1} ---")
                        print("\nSystem Prompt:")
                        print(msgs[0]["content"])
                        print("\nUser Prompt:")
                        print(msgs[1]["content"])
                    if len(batch_question_quality) > 3:
                        print(f"\n... and {len(batch_question_quality) - 3} more samples")
                    print("="*60)
                
                try:
                    responses = judge_llm.batch_inference(messages_list, batch_size=batch_size)
                    
                    if detailed_logging:
                        print(f"\nQUESTION QUALITY JUDGE - BATCH {batch_num} LLM RESPONSES")
                        print("="*60)
                        for j, response in enumerate(responses[:3]):
                            print(f"\n--- Sample {j+1} ---")
                            print(response)
                        if len(responses) > 3:
                            print(f"\n... and {len(responses) - 3} more responses")
                        print("="*60)
                    
                    # Process responses
                    for (idx, _, ground_truth_questions, result), response in zip(batch_question_quality, responses):
                        evaluation = extract_json_from_response(response)
                        
                        if evaluation is None:
                            processed_results[idx]["question_quality_evaluation"] = {
                                "success": False,
                                "error": "Could not parse JSON from judge response",
                                "raw_response": response
                            }
                        else:
                            evaluation["success"] = True
                            processed_results[idx]["question_quality_evaluation"] = evaluation
                            
                            # Compute metrics using calculate_question_score
                            gt_questions_list = parse_ground_truth_questions(ground_truth_questions)
                            if gt_questions_list:
                                try:
                                    score_metrics = calculate_question_score(evaluation, gt_questions_list)
                                    processed_results[idx]["question_quality_metrics"] = score_metrics
                                except Exception as e:
                                    print(f"  Warning: Failed to calculate question score for {result.get('uid')}: {e}")
                            
                            stats["question_quality_evaluations"].append(evaluation)
                
                except Exception as e:
                    print(f"  Error: Batch {batch_num} question quality evaluation failed: {e}")
                    # Fall back to individual evaluation
                    for idx, messages, ground_truth_questions, result in batch_question_quality:
                        try:
                            question_eval = evaluate_question_quality(
                                judge_llm,
                                result.get("misleading_prompt", ""),
                                ground_truth_questions,
                                extract_generated_questions(result),
                                detailed_logging=detailed_logging
                            )
                            processed_results[idx]["question_quality_evaluation"] = question_eval
                            if question_eval.get("success"):
                                gt_questions_list = parse_ground_truth_questions(ground_truth_questions)
                                if gt_questions_list:
                                    try:
                                        score_metrics = calculate_question_score(question_eval, gt_questions_list)
                                        processed_results[idx]["question_quality_metrics"] = score_metrics
                                    except Exception as e:
                                        print(f"  Warning: Failed to calculate question score for {result.get('uid')}: {e}")
                                stats["question_quality_evaluations"].append(question_eval)
                        except Exception as e:
                            print(f"  Warning: Question quality evaluation failed for {result.get('uid')}: {e}")
            
            # Process ambiguity resolution for this batch
            if batch_ambiguity_resolution:
                print(f"\n  Processing {len(batch_ambiguity_resolution)} ambiguity resolution evaluations...")
                messages_list = [msgs for _, msgs, _ in batch_ambiguity_resolution]
                
                if detailed_logging:
                    print("\n" + "="*60)
                    print(f"AMBIGUITY RESOLUTION JUDGE - BATCH {batch_num} LLM REQUEST")
                    print("="*60)
                    for j, (idx, msgs, _) in enumerate(batch_ambiguity_resolution[:3]):
                        print(f"\n--- Sample {idx+1} ---")
                        print("\nSystem Prompt:")
                        print(msgs[0]["content"])
                        print("\nUser Prompt:")
                        print(msgs[1]["content"])
                    if len(batch_ambiguity_resolution) > 3:
                        print(f"\n... and {len(batch_ambiguity_resolution) - 3} more samples")
                    print("="*60)
                
                try:
                    responses = judge_llm.batch_inference(messages_list, batch_size=batch_size)
                    
                    if detailed_logging:
                        print(f"\nAMBIGUITY RESOLUTION JUDGE - BATCH {batch_num} LLM RESPONSES")
                        print("="*60)
                        for j, response in enumerate(responses[:3]):
                            print(f"\n--- Sample {j+1} ---")
                            print(response)
                        if len(responses) > 3:
                            print(f"\n... and {len(responses) - 3} more responses")
                        print("="*60)
                    
                    # Process responses
                    for (idx, _, result), response in zip(batch_ambiguity_resolution, responses):
                        evaluation = extract_json_from_response(response)
                        
                        if evaluation is None:
                            processed_results[idx]["ambiguity_resolution_evaluation"] = {
                                "success": False,
                                "error": "Could not parse JSON from judge response",
                                "raw_response": response
                            }
                        else:
                            evaluation["success"] = True
                            processed_results[idx]["ambiguity_resolution_evaluation"] = evaluation
                            stats["ambiguity_resolution_evaluations"].append(evaluation)
                
                except Exception as e:
                    print(f"  Error: Batch {batch_num} ambiguity resolution evaluation failed: {e}")
                    # Fall back to individual evaluation
                    for idx, messages, result in batch_ambiguity_resolution:
                        try:
                            original_prompt = result.get("original_prompt", "")
                            clarified_prompt = extract_corrected_prompt(result)
                            misleading_prompt = result.get("misleading_prompt", "")
                            what_changed = result.get("what_changed", "")
                            ambiguity_scan = result.get("ambiguity_scan", "")
                            
                            if not what_changed or not ambiguity_scan:
                                step1 = result.get("step1_analysis", {})
                                if not what_changed:
                                    what_changed = step1.get("what_changed", "")
                                if not ambiguity_scan:
                                    ambiguity_scan = step1.get("ambiguity_scan", "")
                            
                            ambiguity_eval = evaluate_ambiguity_resolution(
                                judge_llm,
                                original_prompt,
                                misleading_prompt,
                                clarified_prompt,
                                what_changed=what_changed,
                                ambiguity_scan=ambiguity_scan,
                                detailed_logging=detailed_logging
                            )
                            processed_results[idx]["ambiguity_resolution_evaluation"] = ambiguity_eval
                            if ambiguity_eval.get("success"):
                                stats["ambiguity_resolution_evaluations"].append(ambiguity_eval)
                        except Exception as e:
                            print(f"  Warning: Ambiguity resolution evaluation failed for {result.get('uid')}: {e}")
            
            # Save batch with both scores
            batch_file = _save_batch_results(processed_results, results_dir, judge_model, skip_judge_evaluation, 
                                           stats, results_dir, batch_num=batch_num, batch_indices=batch_indices)
            print(f"\n  ✓ Saved batch {batch_num} to {batch_file}")
    
    # Update stats with existing evaluations (from resume)
    for processed_result in processed_results:
        existing_question_eval = processed_result.get("question_quality_evaluation", {})
        existing_ambiguity_eval = processed_result.get("ambiguity_resolution_evaluation", {})
        
        if existing_question_eval.get("success"):
            if existing_question_eval not in stats["question_quality_evaluations"]:
                stats["question_quality_evaluations"].append(existing_question_eval)
        
        if existing_ambiguity_eval.get("success"):
            if existing_ambiguity_eval not in stats["ambiguity_resolution_evaluations"]:
                stats["ambiguity_resolution_evaluations"].append(existing_ambiguity_eval)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    
    # Success rate
    success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
    
    # Chamfer Distance statistics
    cd_mean = np.mean(stats["chamfer_distances"]) if stats["chamfer_distances"] else None
    cd_median = np.median(stats["chamfer_distances"]) if stats["chamfer_distances"] else None
    
    # Question quality statistics (using calculate_question_score metrics)
    question_quality_stats = {
        "total_evaluated": 0,
        "total_matched": 0,
        "total_hallucinated": 0,
        "match_rates": [],  # match_rate per sample (matched_questions / total_generated)
        "matched_questions": [],
        "hallucinated_questions": [],
        "f1_scores": [],
        "precision_scores": [],
        "recall_scores": []
    }
    
    for processed_result in processed_results:
        eval_result = processed_result.get("question_quality_evaluation", {})
        metrics = processed_result.get("question_quality_metrics", {})
        
        if eval_result.get("success"):
            matched_questions = eval_result.get("matched_questions", [])
            hallucinated_questions = eval_result.get("hallucinated_questions", [])
            
            total_generated = len(matched_questions) + len(hallucinated_questions)
            num_matched = len(matched_questions)
            num_hallucinated = len(hallucinated_questions)
            
            if total_generated > 0:
                question_quality_stats["total_evaluated"] += total_generated
                question_quality_stats["total_matched"] += num_matched
                question_quality_stats["total_hallucinated"] += num_hallucinated
                match_rate = num_matched / total_generated
                question_quality_stats["match_rates"].append(match_rate)
                
                question_quality_stats["matched_questions"].extend(matched_questions)
                question_quality_stats["hallucinated_questions"].extend(hallucinated_questions)
            
            # Add metrics from calculate_question_score if available
            if metrics:
                if "score" in metrics:  # F1 score
                    question_quality_stats["f1_scores"].append(metrics["score"])
                if "precision" in metrics:
                    question_quality_stats["precision_scores"].append(metrics["precision"])
                if "recall" in metrics:
                    question_quality_stats["recall_scores"].append(metrics["recall"])
    
    # Ambiguity resolution statistics
    ambiguity_resolution_scores = []
    ambiguity_resolution_reasonings = []
    resolution_distribution = {"1.0": 0, "0.5": 0, "0.0": 0}
    
    for eval_result in stats["ambiguity_resolution_evaluations"]:
        if eval_result.get("success"):
            score = eval_result.get("score")
            reasoning = eval_result.get("reasoning", "")
            
            if score is not None:
                ambiguity_resolution_scores.append(float(score))
                ambiguity_resolution_reasonings.append(reasoning)
                
                # Track distribution
                if score == 1.0:
                    resolution_distribution["1.0"] += 1
                elif score == 0.5:
                    resolution_distribution["0.5"] += 1
                elif score == 0.0:
                    resolution_distribution["0.0"] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Samples: {stats['total']}")
    print(f"\n--- Success Rate ---")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Successful: {stats['success']}/{stats['total']}")
    print(f"  Failed: {stats['failed']}/{stats['total']}")
    print(f"  Skipped: {stats['skipped']}/{stats['total']}")
    
    if cd_mean is not None:
        print(f"\n--- Chamfer Distance ---")
        print(f"  Mean: {cd_mean:.6f}")
        print(f"  Median: {cd_median:.6f}")
        print(f"  Min: {np.min(stats['chamfer_distances']):.6f}")
        print(f"  Max: {np.max(stats['chamfer_distances']):.6f}")
        print(f"  Std: {np.std(stats['chamfer_distances']):.6f}")
        print(f"  Count: {len(stats['chamfer_distances'])}")
    
    if stats["question_quality_evaluations"]:
        print(f"\n--- Question Quality (Judge LLM) ---")
        print(f"  Samples Evaluated: {len(stats['question_quality_evaluations'])}")
        print(f"  Total Questions Evaluated: {question_quality_stats['total_evaluated']}")
        print(f"  Total Matched: {question_quality_stats['total_matched']}")
        print(f"  Total Hallucinated: {question_quality_stats['total_hallucinated']}")
        if question_quality_stats['total_evaluated'] > 0:
            overall_match_rate = question_quality_stats['total_matched'] / question_quality_stats['total_evaluated']
            print(f"  Overall Match Rate: {overall_match_rate:.2%}")
        if question_quality_stats['match_rates']:
            print(f"  Mean Match Rate per Sample: {np.mean(question_quality_stats['match_rates']):.2%}")
            print(f"  Median Match Rate per Sample: {np.median(question_quality_stats['match_rates']):.2%}")
        if question_quality_stats['f1_scores']:
            print(f"  F1 Score: Mean={np.mean(question_quality_stats['f1_scores']):.3f}, Median={np.median(question_quality_stats['f1_scores']):.3f} (n={len(question_quality_stats['f1_scores'])})")
            print(f"  Precision: Mean={np.mean(question_quality_stats['precision_scores']):.3f}, Median={np.median(question_quality_stats['precision_scores']):.3f} (n={len(question_quality_stats['precision_scores'])})")
            print(f"  Recall: Mean={np.mean(question_quality_stats['recall_scores']):.3f}, Median={np.median(question_quality_stats['recall_scores']):.3f} (n={len(question_quality_stats['recall_scores'])})")
    
    if stats["ambiguity_resolution_evaluations"]:
        print(f"\n--- Ambiguity Resolution (Judge LLM) ---")
        print(f"  Samples Evaluated: {len(stats['ambiguity_resolution_evaluations'])}")
        if ambiguity_resolution_scores:
            print(f"  Mean Score: {np.mean(ambiguity_resolution_scores):.2f}")
            print(f"  Median Score: {np.median(ambiguity_resolution_scores):.2f}")
            print(f"  Distribution:")
            print(f"    Fully Resolved (1.0): {resolution_distribution['1.0']} ({resolution_distribution['1.0']/len(ambiguity_resolution_scores)*100:.1f}%)")
            print(f"    Partially Resolved (0.5): {resolution_distribution['0.5']} ({resolution_distribution['0.5']/len(ambiguity_resolution_scores)*100:.1f}%)")
            print(f"    Unresolved (0.0): {resolution_distribution['0.0']} ({resolution_distribution['0.0']/len(ambiguity_resolution_scores)*100:.1f}%)")
    
    print(f"\nResults saved to batch files in: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process clarification pipeline results")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory containing clarification_result_*.json files (batch files will be saved here)")
    parser.add_argument("--single_file", type=str, default=None,
                       help="Process a single clarification result file (overrides results_dir)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                       help=f"Model for judge LLM (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--skip_judge", action="store_true",
                       help="Skip judge LLM evaluation (only calculate basic stats)")
    parser.add_argument("--detailed_logging", action="store_true",
                       help="Enable detailed logging (show full LLM messages and responses)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Limit processing to first N samples")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Batch size for judge LLM inference (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--source_json", type=str, default=None,
                       help="Path to source JSON file (selected_misleading_samples_ratio10.json) to load what_changed and ambiguity_scan fields")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single_file and not args.results_dir:
        parser.error("Either --results_dir or --single_file must be provided")
    
    if args.single_file:
        # Use directory of single file as results_dir
        results_dir = os.path.dirname(args.single_file) or "."
    else:
        results_dir = args.results_dir
    
    main(
        results_dir=results_dir,
        judge_model=args.judge_model,
        skip_judge_evaluation=args.skip_judge,
        detailed_logging=args.detailed_logging,
        n_samples=args.n_samples,
        single_file=args.single_file,
        batch_size=args.batch_size,
        source_json_path=args.source_json
    )
