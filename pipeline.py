"""
Parallel Clarification Pipeline for CAD Agent.

This version processes samples in batches and performs batch inference for each step.
Handles early stopping when samples are skipped (not misleading, no questions, etc.).
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import copy

from src.inference import LLM
from src.ask_agent import AskAgent
from src.mesh_utils import cadquery_to_mesh, load_mesh_from_stl, normalize_mesh, mesh_to_pointcloud
from src.evaluation import chamfer_distance
from config.clarification import (
    ANSWER_QUESTIONS_PROMPT_TEMPLATE,
    CLARIFY_WITH_ANSWERS_PROMPT_TEMPLATE
)
from config.code_generation import (
    CODE_GENERATION_SYSTEM_PROMPT,
    CODE_GENERATION_USER_PROMPT_TEMPLATE
)

# =============================================================================
# Configuration (override via environment variables). See README for details.
# =============================================================================

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
DEFAULT_MISLEADING_JSON = os.environ.get(
    "MISLEADING_JSON", "./dataset/selected_misleading_samples_test"
)
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./clarification_results")
GT_MESH_DIR = os.environ.get(
    "GT_MESH_DIR", os.path.join(DATA_ROOT, "text2cad/deepcad_mesh")
)

CLARIFY_AGENT_MODEL = os.environ.get("CLARIFY_AGENT_MODEL", "gpt-4o-mini-2024-07-18")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "gpt-5-mini-2025-08-07")
CODE_GEN_MODEL = os.environ.get("CODE_GEN_MODEL", "BBexist/ProCAD-coder")


# =============================================================================
# Helpers
# =============================================================================

def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def evaluate_code(code: str, uid: str, gt_mesh_dir: str = GT_MESH_DIR, n_points: int = 8192) -> Dict:
    """Evaluate generated CadQuery code against a ground-truth STL via Chamfer distance."""
    result = {'success': False, 'chamfer_distance': None, 'error': None}

    gt_mesh_path = Path(gt_mesh_dir) / f"{uid}.stl"
    if not gt_mesh_path.exists():
        result['error'] = f"Ground truth mesh not found: {gt_mesh_path}"
        return result

    try:
        gt_mesh = load_mesh_from_stl(str(gt_mesh_path))
        gt_points = mesh_to_pointcloud(normalize_mesh(gt_mesh), n_points)

        gen_mesh = cadquery_to_mesh(code)
        gen_points = mesh_to_pointcloud(normalize_mesh(gen_mesh), n_points)

        cd = chamfer_distance(gt_points, gen_points)
        result['success'] = True
        result['chamfer_distance'] = cd
        if cd < 0.0001:
            result['quality'] = 'excellent'
        elif cd < 0.0005:
            result['quality'] = 'good'
        elif cd < 0.001:
            result['quality'] = 'acceptable'
        else:
            result['quality'] = 'poor'
    except Exception as e:
        result['error'] = str(e)

    return result


def load_all_samples_from_json(json_path: str) -> List[Dict]:
    """Load all samples from a misleading-samples JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("samples", [])


def load_processed_samples(output_dir: str) -> set:
    """
    Load (uid, k, config_name) tuples that have already been processed from existing files.
    Checks both batch files and individual result files.
    Uses (uid, k, config_name) to distinguish different misleading types for the same (uid, k).
    
    For backward compatibility with Claude 4.5 results:
    - If config_name is missing and k > 0, infer config_name from k: k=1 -> "underspec_k1", k=2 -> "underspec_k2"
    - If config_name is missing and k=0, use None (original correct samples)
    
    Args:
        output_dir: Directory containing clarification_batch_*.json or clarification_result_*.json files
        
    Returns:
        Set of already processed (uid, k, config_name) tuples
    """
    processed_samples = set()
    
    # Check batch files (new format)
    batch_files = glob.glob(os.path.join(output_dir, 'clarification_batch_*.json'))
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r') as f:
                data = json.load(f)
            for result in data.get('results', []):
                uid = result.get('uid')
                k = result.get('k')
                config_name = result.get('config_name')
                
                # Backward compatibility: infer config_name from k for Claude 4.5 results
                if uid and config_name is None and k is not None:
                    if k == 1:
                        config_name = "underspec_k1"
                    elif k == 2:
                        config_name = "underspec_k2"
                    # k=0 or other values keep config_name as None
                
                if uid:
                    processed_samples.add((uid, k, config_name))
        except Exception as e:
            print(f"Warning: Failed to read {batch_file}: {e}")
    
    # Check individual result files (old format: clarification_result_{uid}_k{k}.json)
    result_files = glob.glob(os.path.join(output_dir, 'clarification_result_*.json'))
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            uid = data.get('uid')
            k = data.get('k')
            config_name = data.get('config_name')
            
            # Backward compatibility: infer config_name from k for Claude 4.5 results
            if uid and config_name is None and k is not None:
                if k == 1:
                    config_name = "underspec_k1"
                elif k == 2:
                    config_name = "underspec_k2"
                # k=0 or other values keep config_name as None
            
            if uid:
                processed_samples.add((uid, k, config_name))
        except Exception as e:
            print(f"Warning: Failed to read {result_file}: {e}")
    
    return processed_samples


class ParallelClarificationPipeline:
    """
    Parallel version of ClarificationPipeline that processes samples in batches.
    """
    
    def __init__(
        self,
        clarify_agent_model: str = CLARIFY_AGENT_MODEL,
        answer_model: str = ANSWER_MODEL,
        code_gen_model: str = CODE_GEN_MODEL,
        batch_size: int = 8
    ):
        """Initialize the pipeline with models for each stage."""
        print("Initializing Parallel Clarification Pipeline...")
        
        # Stage 1 & 3: ClarifyAgent (detect misleading + generate questions + generate corrected description)
        print(f"  Loading ClarifyAgent: {clarify_agent_model}")
        self.clarify_agent = AskAgent(model_name=clarify_agent_model)
        
        # Stage 2: Answer questions
        print(f"  Loading Answer Model: {answer_model}")
        self.answer_llm = LLM(model_name=answer_model)
        
        # Stage 4: Code generation
        print(f"  Loading Code Gen Model: {code_gen_model}")
        self.code_gen_llm = LLM(model_name=code_gen_model)
        
        self.batch_size = batch_size
        print(f"  Batch size: {batch_size}")
        print("Pipeline initialized!")
    
    def process_batch(
        self,
        samples: List[Dict],
        misleading_json_path: Optional[str] = None,
        skip_if_not_misleading: bool = True,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Process a batch of samples with parallel inference at each step.
        
        Args:
            samples: List of sample dicts with uid, original_prompt, misleading_prompt, etc.
            misleading_json_path: Path to JSON file (for loading additional fields)
            skip_if_not_misleading: If True, skip clarification if prompt seems clear
            verbose: If True, print progress logs
            
        Returns:
            List of result dicts (one per sample)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing batch of {len(samples)} samples")
            print(f"{'='*80}")
        
        # Initialize results for all samples
        results = []
        for sample in samples:
            uid = sample.get("uid")
            original_prompt = sample.get("original_prompt")
            misleading_prompt = sample.get("misleading_description") or sample.get("misleading_prompt")
            k = sample.get("k", "unknown")
            
            result = {
                "uid": uid,
                "original_prompt": original_prompt,
                "misleading_prompt": misleading_prompt,
                "ground_truth_questions": sample.get("questions_to_ask"),
                "ground_truth_answers": sample.get("answer_to_questions"),
                "what_changed": sample.get("what_changed", ""),
                "ambiguity_scan": sample.get("ambiguity_scan", ""),
                "step1_analysis": None,
                "step2_answers": None,
                "step3_corrected_description": None,
                "step4_code": None,
                "step4_code_extracted": None,
                "evaluation": None,
                "evaluation_original": None,
                "evaluation_misleading": None,
                "skipped": False,
                "skip_reason": None,
                "k": k,
                "config_name": sample.get("config_name"),  # Store config_name to distinguish misleading types
                "original_cd_from_json": sample.get("original_cd"),
                "misleading_cd_from_json": sample.get("misleading_cd")
            }
            results.append(result)
        
        # Step 1: Batch detect and ask questions
        if verbose:
            print("\nStep 1: Batch detecting misleading prompts and generating questions...")
        
        step1_messages_list = []
        active_indices = list(range(len(samples)))
        
        for i in active_indices:
            sample = samples[i]
            misleading_prompt = sample.get("misleading_description") or sample.get("misleading_prompt")
            # For k=0 samples, misleading_prompt is None, so use original_prompt instead
            if misleading_prompt is None:
                misleading_prompt = sample.get("original_prompt")
            messages = self.clarify_agent._build_messages(misleading_prompt)
            step1_messages_list.append(messages)
        
        # Batch inference for step1
        step1_responses = self.clarify_agent.llm.batch_inference(
            step1_messages_list,
            batch_size=self.batch_size,
            max_new_tokens=self.clarify_agent.max_new_tokens,
            temperature=self.clarify_agent.temperature
        )
        
        # Parse step1 responses
        step1_analyses = []
        step1_messages_with_response = []
        for i, (response, messages) in enumerate(zip(step1_responses, step1_messages_list)):
            analysis = self.clarify_agent._parse_response(response)
            analysis["original_prompt"] = samples[i].get("original_prompt")
            analysis["raw_response"] = response
            step1_analyses.append(analysis)
            
            messages_with_response = messages + [{"role": "assistant", "content": response}]
            step1_messages_with_response.append(messages_with_response)
            
            results[i]["step1_analysis"] = analysis
        
        # Filter active samples based on step1 results
        new_active_indices = []
        for i in active_indices:
            analysis = step1_analyses[i]
            is_misleading = analysis.get("is_misleading", False)
            questions = analysis.get("questions", [])
            
            if not is_misleading and skip_if_not_misleading:
                results[i]["skipped"] = True
                results[i]["skip_reason"] = "Prompt detected as clear/not misleading"
                # Handle evaluation for skipped samples
                self._handle_skipped_evaluation(results[i], samples[i], verbose)
            elif not questions:
                results[i]["skipped"] = True
                results[i]["skip_reason"] = "No clarifying questions generated"
                self._handle_skipped_evaluation(results[i], samples[i], verbose)
            else:
                new_active_indices.append(i)
        
        active_indices = new_active_indices
        
        if verbose:
            print(f"  Active after step1: {len(active_indices)}/{len(samples)}")
        
        # Step 2: Batch answer questions
        if active_indices:
            if verbose:
                print(f"\nStep 2: Batch answering questions for {len(active_indices)} samples...")
            
            step2_messages_list = []
            step2_indices = []
            
            for i in active_indices:
                sample = samples[i]
                original_prompt = sample.get("original_prompt")
                misleading_prompt = sample.get("misleading_description") or sample.get("misleading_prompt")
                # For k=0 samples, misleading_prompt is None, so use original_prompt instead
                if misleading_prompt is None:
                    misleading_prompt = original_prompt
                questions = step1_analyses[i].get("questions", [])
                
                if questions:
                    questions_str = "\n".join([f"{j+1}. {q}" for j, q in enumerate(questions)])
                    prompt = ANSWER_QUESTIONS_PROMPT_TEMPLATE.format(
                        original_prompt=original_prompt,
                        misleading_prompt=misleading_prompt,
                        questions=questions_str
                    )
                    messages = [{"role": "user", "content": prompt}]
                    step2_messages_list.append(messages)
                    step2_indices.append(i)
            
            # Batch inference for step2
            if step2_messages_list:
                step2_responses = self.answer_llm.batch_inference(
                    step2_messages_list,
                    batch_size=self.batch_size,
                    max_new_tokens=1024
                )
                
                # Store step2 results
                for idx, (response, orig_idx) in enumerate(zip(step2_responses, step2_indices)):
                    results[orig_idx]["step2_answers"] = response
        
        # Step 3: Batch generate corrected descriptions
        if active_indices:
            if verbose:
                print(f"\nStep 3: Batch generating corrected descriptions for {len(active_indices)} samples...")
            
            step3_messages_list = []
            step3_indices = []
            
            for i in active_indices:
                sample = samples[i]
                answers = results[i].get("step2_answers")
                if answers:
                    # Use previous messages from step1 + answers
                    previous_messages = step1_messages_with_response[i]
                    messages = previous_messages + [{"role": "user", "content": answers}]
                    step3_messages_list.append(messages)
                    step3_indices.append(i)
            
            # Batch inference for step3
            if step3_messages_list:
                step3_responses = self.clarify_agent.llm.batch_inference(
                    step3_messages_list,
                    batch_size=self.batch_size,
                    max_new_tokens=1024
                )
                
                # Store step3 results and extract standardized_prompt
                for idx, (response, orig_idx) in enumerate(zip(step3_responses, step3_indices)):
                    results[orig_idx]["step3_corrected_description"] = response
                    
                    # Extract standardized_prompt
                    standardized_prompt = self._extract_standardized_prompt(response)
                    results[orig_idx]["_standardized_prompt"] = standardized_prompt
        
        # Step 4: Batch generate code
        if active_indices:
            if verbose:
                print(f"\nStep 4: Batch generating code for {len(active_indices)} samples...")
            
            step4_messages_list = []
            step4_indices = []
            
            for i in active_indices:
                standardized_prompt = results[i].get("_standardized_prompt")
                original_prompt = samples[i].get("original_prompt")
                
                # Check if standardized matches original
                if standardized_prompt and standardized_prompt.strip() == original_prompt.strip():
                    # Skip code generation, use original_cd
                    results[i]["evaluation"] = {
                        "success": True,
                        "chamfer_distance": samples[i].get("original_cd"),
                        "quality": "N/A"
                    }
                    results[i]["step4_code"] = None
                    results[i]["step4_code_extracted"] = None
                elif standardized_prompt:
                    # Generate code
                    messages = [
                        {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": CODE_GENERATION_USER_PROMPT_TEMPLATE.format(description=standardized_prompt)}
                    ]
                    step4_messages_list.append(messages)
                    step4_indices.append(i)
            
            # Batch inference for step4
            if step4_messages_list:
                step4_responses = self.code_gen_llm.batch_inference(
                    step4_messages_list,
                    batch_size=self.batch_size,
                    max_new_tokens=4096
                )
                
                # Store step4 results and evaluate
                for idx, (response, orig_idx) in enumerate(zip(step4_responses, step4_indices)):
                    results[orig_idx]["step4_code"] = response
                    results[orig_idx]["step4_code_extracted"] = extract_code(response)
                    
                    # Evaluate code
                    uid = results[orig_idx]["uid"]
                    eval_result = evaluate_code(results[orig_idx]["step4_code_extracted"], uid)
                    results[orig_idx]["evaluation"] = eval_result
        
        # Set evaluation_original and evaluation_misleading from JSON
        for i, result in enumerate(results):
            sample = samples[i]
            result["evaluation_original"] = {
                "success": True,
                "chamfer_distance": sample.get("original_cd"),
                "quality": "N/A"
            }
            result["evaluation_misleading"] = {
                "success": sample.get("misleading_success", False),
                "chamfer_distance": sample.get("misleading_cd"),
                "quality": "N/A"
            }
        
        return results
    
    def _extract_standardized_prompt(self, response: str) -> Optional[str]:
        """Extract standardized_prompt from step3 JSON response."""
        import re
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_response, re.DOTALL)
                if match:
                    cleaned_response = match.group(1)
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"standardized_prompt"[^}]*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed.get("standardized_prompt")
            else:
                parsed = json.loads(cleaned_response)
                return parsed.get("standardized_prompt")
        except Exception:
            return response  # Fallback to raw response
    
    def _handle_skipped_evaluation(self, result: Dict, sample: Dict, verbose: bool):
        """Handle evaluation for skipped samples."""
        if sample.get("original_cd") is not None:
            result["evaluation_original"] = {
                "success": True,
                "chamfer_distance": sample.get("original_cd"),
                "quality": "N/A"
            }
            result["evaluation_misleading"] = {
                "success": sample.get("misleading_success", False),
                "chamfer_distance": sample.get("misleading_cd"),
                "quality": "N/A"
            }
            result["evaluation"] = result["evaluation_misleading"]


def process_all_misleading_samples_parallel(
    json_path: str = DEFAULT_MISLEADING_JSON,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 8,
    inference_batch_size: int = 8,
    verbose: bool = True,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    clarify_agent_model: Optional[str] = None,
    answer_model: Optional[str] = None,
    code_gen_model: Optional[str] = None,
):
    """
    Process all misleading samples from JSON file in parallel batches.
    
    Args:
        json_path: Path to JSON file with misleading samples
        output_dir: Directory to save results
        batch_size: Number of samples to process in each batch
        inference_batch_size: Batch size for LLM inference within each step
        verbose: If True, print progress logs
        start_idx: Start processing from this index (default: 0)
        end_idx: Stop processing at this index (default: None = all)
        clarify_agent_model: Model for ask/clarify agent
        answer_model: Model for answer agent
        code_gen_model: Model for code generation
    """
    # Load all samples
    if verbose:
        print(f"Loading samples from {json_path}...")
    all_samples = load_all_samples_from_json(json_path)
    
    if end_idx is None:
        end_idx = len(all_samples)
    
    samples_to_process = all_samples[start_idx:end_idx]
    
    # Check for already processed samples (resume support)
    os.makedirs(output_dir, exist_ok=True)
    processed_samples = load_processed_samples(output_dir)
    if processed_samples:
        original_count = len(samples_to_process)
        # Filter by (uid, k, config_name) to distinguish different misleading types
        # For backward compatibility: if config_name is None in processed samples, 
        # also check (uid, k, None) for samples without config_name
        original_samples = samples_to_process
        samples_to_process = []
        skipped_count = 0
        for s in original_samples:
            uid = s.get('uid')
            k = s.get('k')
            config_name = s.get('config_name')
            sample_key = (uid, k, config_name)
            
            # Check if this exact (uid, k, config_name) is already processed
            if sample_key in processed_samples:
                skipped_count += 1
                continue
            
            # Backward compatibility: if old results don't have config_name,
            # also skip if (uid, k, None) exists in processed_samples
            if (uid, k, None) in processed_samples and config_name is not None:
                # This means an old result exists without config_name
                # We'll still process this one since it has a different config_name
                pass
            
            samples_to_process.append(s)
        
        skipped = original_count - len(samples_to_process)
        if verbose:
            print(f"Resume: Found {len(processed_samples)} already processed (uid, k, config_name) tuples, skipping {skipped} samples")
    
    if len(samples_to_process) == 0:
        if verbose:
            print("All samples already processed. Nothing to do.")
        return
    
    if verbose:
        print(f"Processing {len(samples_to_process)} samples (indices {start_idx} to {end_idx-1})")
        print(f"Batch size: {batch_size}, Inference batch size: {inference_batch_size}")
        print(f"Models:")
        print(f"  Clarify Agent: {clarify_agent_model or CLARIFY_AGENT_MODEL}")
        print(f"  Answer Agent: {answer_model or ANSWER_MODEL}")
        print(f"  Code Gen: {code_gen_model or CODE_GEN_MODEL}")
    
    # Initialize pipeline
    pipeline = ParallelClarificationPipeline(
        clarify_agent_model=clarify_agent_model or CLARIFY_AGENT_MODEL,
        answer_model=answer_model or ANSWER_MODEL,
        code_gen_model=code_gen_model or CODE_GEN_MODEL,
        batch_size=inference_batch_size
    )
    
    # Find starting batch number for resume
    existing_batches = glob.glob(os.path.join(output_dir, 'clarification_batch_*.json'))
    start_batch_num = len(existing_batches) + 1
    
    # Process in batches
    all_results = []
    batch_num = 0
    current_run_processed = set(processed_samples)  # Track (uid, k, config_name) tuples processed in this run
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "clarified": 0,
        "cds": [],
        "cds_original": [],
        "cds_misleading": [],
        "cds_clarified": []
    }
    
    for batch_idx in range(0, len(samples_to_process), batch_size):
        batch_end = min(batch_idx + batch_size, len(samples_to_process))
        batch_samples = samples_to_process[batch_idx:batch_end]
        batch_num = start_batch_num + (batch_idx // batch_size)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing batch {batch_num} (samples {batch_idx} to {batch_end-1})")
            print(f"{'='*80}")
        
        try:
            batch_results = pipeline.process_batch(
                batch_samples,
                misleading_json_path=json_path,
                skip_if_not_misleading=True,
                verbose=verbose
            )
            
            # Safety check: Filter out any (uid, k, config_name) tuples that were already processed in this run
            filtered_batch_results = []
            skipped_duplicates = 0
            for result in batch_results:
                uid = result.get("uid")
                k = result.get("k")
                config_name = result.get("config_name")
                sample_key = (uid, k, config_name)
                if uid and sample_key not in current_run_processed:
                    filtered_batch_results.append(result)
                    current_run_processed.add(sample_key)  # Track in current run
                elif uid:
                    skipped_duplicates += 1
            
            if skipped_duplicates > 0 and verbose:
                print(f"  WARNING: Skipped {skipped_duplicates} duplicate (uid, k, config_name) tuples (already processed in this run)")
            
            all_results.extend(filtered_batch_results)
            
            # Update stats
            for result in filtered_batch_results:
                stats["total"] += 1
                if result.get("skipped"):
                    stats["skipped"] += 1
                else:
                    stats["clarified"] += 1
                
                eval_result = result.get("evaluation", {})
                if eval_result and eval_result.get("success"):
                    stats["success"] += 1
                    stats["cds_clarified"].append(eval_result.get("chamfer_distance"))
                
                eval_original = result.get("evaluation_original", {})
                if eval_original and eval_original.get("success"):
                    stats["cds_original"].append(eval_original.get("chamfer_distance"))
                
                eval_misleading = result.get("evaluation_misleading", {})
                if eval_misleading and eval_misleading.get("success"):
                    stats["cds_misleading"].append(eval_misleading.get("chamfer_distance"))
            
            # Save batch results
            batch_output_file = os.path.join(output_dir, f"clarification_batch_{batch_num:04d}.json")
            with open(batch_output_file, 'w') as f:
                json.dump({
                    "batch_number": batch_num,
                    "batch_start": batch_idx,
                    "batch_end": batch_end,
                    "timestamp": datetime.now().isoformat(),
                    "count": len(filtered_batch_results),
                    "results": filtered_batch_results
                }, f, indent=2)
            
            if verbose:
                print(f"\nSaved batch {batch_num} to {batch_output_file}")
        
        except Exception as e:
            print(f"ERROR processing batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            stats["failed"] += len(batch_samples)
    
    # Save all results
    all_results_file = os.path.join(output_dir, f"all_results_parallel_{start_idx}_{end_idx}.json")
    with open(all_results_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_processed": len(all_results),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size,
                "inference_batch_size": inference_batch_size,
                "models": {
                    "clarify_agent": clarify_agent_model or CLARIFY_AGENT_MODEL,
                    "answer_model": answer_model or ANSWER_MODEL,
                    "code_gen_model": code_gen_model or CODE_GEN_MODEL
                },
                "json_path": json_path
            },
            "stats": stats,
            "results": all_results
        }, f, indent=2)
    
    if verbose:
        print(f"\nAll results saved to: {all_results_file}")
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Samples: {stats['total']}")
        print(f"Successful Evaluations: {stats['success']}")
        print(f"Failed Evaluations: {stats['failed']}")
        print(f"Skipped (not misleading): {stats['skipped']}")
        print(f"Clarified: {stats['clarified']}")
        
        import numpy as np
        if stats["cds_original"]:
            cds_orig = np.array(stats["cds_original"])
            print(f"\n--- ORIGINAL Prompt CD (Baseline) ---")
            print(f"  Mean: {np.mean(cds_orig):.6f}")
        
        if stats["cds_misleading"]:
            cds_misl = np.array(stats["cds_misleading"])
            print(f"\n--- MISLEADING Prompt CD (No Clarification) ---")
            print(f"  Mean: {np.mean(cds_misl):.6f}")
        
        if stats["cds_clarified"]:
            cds = np.array(stats["cds_clarified"])
            print(f"\n--- CLARIFIED Prompt CD (After Q&A) ---")
            print(f"  Mean: {np.mean(cds):.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Clarification Pipeline")
    parser.add_argument("--json_path", type=str, default=DEFAULT_MISLEADING_JSON,
                       help="Path to JSON file with misleading samples")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of samples to process in each batch")
    parser.add_argument("--inference_batch_size", type=int, default=8,
                       help="Batch size for LLM inference within each step")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="End index for processing (None = all)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    parser.add_argument("--clarify_model", type=str, default=None,
                       help=f"Model for clarify/ask agent (default: {CLARIFY_AGENT_MODEL})")
    parser.add_argument("--answer_model", type=str, default=None,
                       help=f"Model for answer agent (default: {ANSWER_MODEL})")
    parser.add_argument("--code_model", type=str, default=None,
                       help=f"Model for code generation (default: {CODE_GEN_MODEL})")
    
    args = parser.parse_args()
    
    process_all_misleading_samples_parallel(
        json_path=args.json_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        verbose=args.verbose,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        clarify_agent_model=args.clarify_model,
        answer_model=args.answer_model,
        code_gen_model=args.code_model
    )
