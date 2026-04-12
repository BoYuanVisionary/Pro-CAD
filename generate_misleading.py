"""
Batch Misleading Prompt Generation for CAD Agent.

Generates misleading CAD descriptions in batches using async OpenAI API.
Supports two ambiguity families (pick with --type):
  - underspec : ambiguity_under_specified (missing / under-defined parameters)
  - conflict  : direct_conflict_same_feature_two_values (contradictory values)
"""

import json
import os
import re
import glob
import random
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config.misleading_prompt import Misleading_system_prompt
from config.ambiguity_under_specified import FEWSHOT_EXAMPLES_UNDERSPEC_REMOVE_ONE_DIM
from config.direct_conflict_same_feature_two_values import FEWSHOT_EXAMPLES_CONFLICTED_DUPLICATE_DIM


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BATCH_DATA_DIR = os.environ.get("BATCH_DATA_DIR", "./sft/filter_data")
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_N_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_CD_THRESHOLD = 0.0002

# Set OPENAI_API_KEY in your environment before running.

# Ambiguity-type descriptions (sent to the generator LLM)
AMBIGUITY_UNDERSPEC_DESCRIPTION = """ambiguity_under_specified:
- The description omits or under-defines a critical parameter (e.g., a dimension, angle, count, or position).
- This is missing information that the user forgot to include, not information implied by context.
- Examples: "a rectangle" without width or height; "an array of holes" without count or spacing; "a fillet" without radius."""

DIRECT_CONFLICT_DESCRIPTION = """direct_conflict_same_feature_two_values:
- Logical conflict: you assert a constraint, then later give explicit incompatible values for the same feature.
- Examples:
- \"Top edge length is 200.\" later: \"Top edge length is 180.\"
- \"Cut hole radius is 52.\" later: \"Use radius 50 for the same through-cut.\"
- \"Extrude 200 in the negative normal.\" later: \"Tube extends +200 along the workplane normal.\""""

# Per-type config: each entry defines the ambiguity identifier, human-readable
# description, few-shot examples, config-name prefix, and default output dir.
TYPE_CONFIG = {
    "underspec": {
        "ambiguity_type": "ambiguity_under_specified",
        "description": AMBIGUITY_UNDERSPEC_DESCRIPTION,
        "fewshot": FEWSHOT_EXAMPLES_UNDERSPEC_REMOVE_ONE_DIM,
        "config_prefix": "underspec",
        "default_output_dir": "./sft/misleading_data",
    },
    "conflict": {
        "ambiguity_type": "direct_conflict_same_feature_two_values",
        "description": DIRECT_CONFLICT_DESCRIPTION,
        "fewshot": FEWSHOT_EXAMPLES_CONFLICTED_DUPLICATE_DIM,
        "config_prefix": "conflict",
        "default_output_dir": "./sft/misleading_data_type2",
    },
}


def build_batch_configs(type_key: str, k_values: Optional[List[int]] = None) -> List[Dict]:
    """Build batch config list for a given ambiguity type and K values."""
    cfg = TYPE_CONFIG[type_key]
    ks = k_values if k_values else [1, 2, 3]
    return [
        {
            "ambiguity_types": [cfg["ambiguity_type"]],
            "k": k,
            "config_name": f"{cfg['config_prefix']}_k{k}",
        }
        for k in ks
    ]


# Path to train/test split file
DEFAULT_UID_FILE = os.environ.get("UID_FILE", "./dataset/train_val_test.json")


def format_fewshot_examples(fewshot_examples: List[Dict]) -> str:
    """Format few-shot examples for the prompt."""
    examples_text = []
    for i, example in enumerate(fewshot_examples, 1):
        inp = example["input"]
        out = example["output"]
        examples_text.append(f"""--- Example {i} ---
RIGHT_PROMPT:
{inp["RIGHT_PROMPT"]}

AMBIGUITY_TYPES: {inp["AMBIGUITY_TYPES"]}
NUM_AMBIGUITIES: {inp["NUM_AMBIGUITIES"]}

Expected Output:
{out}
""")
    return "\n".join(examples_text)


# =============================================================================
# Utility Functions
# =============================================================================

def format_ambiguity_types_for_prompt(ambiguity_types: List[str]) -> str:
    """Return the human-readable description string for the given ambiguity type."""
    for cfg in TYPE_CONFIG.values():
        if cfg["ambiguity_type"] in ambiguity_types:
            return cfg["description"]
    return ""


def parse_misleading_response(response: str) -> Dict:
    """Parse the structured response from the misleading prompt generator."""
    result = {
        "misleading_description": "",
        "what_changed": "",
        "ambiguity_scan": "",
        "questions_to_ask": "",
        "answer_to_questions": ""
    }
    
    sections = [
        ("MISLEADING_DESCRIPTION", "misleading_description"),
        ("WHAT_I_CHANGED", "what_changed"),
        ("AMBIGUITY SCAN", "ambiguity_scan"),
        ("QUESTIONS_TO_ASK", "questions_to_ask"),
        ("ANSWER_TO_QUESTIONS", "answer_to_questions")
    ]
    
    text = response
    
    for i, (marker, key) in enumerate(sections):
        start_patterns = [f"1) {marker}", f"2) {marker}", f"3) {marker}", f"4) {marker}",
                         f"{marker}:", f"{marker}\n", marker]
        
        start_idx = -1
        for pattern in start_patterns:
            idx = text.find(pattern)
            if idx != -1:
                start_idx = idx + len(pattern)
                break
        
        if start_idx != -1:
            end_idx = len(text)
            for next_marker, _ in sections[i+1:]:
                for pattern in [f"1) {next_marker}", f"2) {next_marker}", 
                               f"3) {next_marker}", f"4) {next_marker}",
                               f"{next_marker}:", f"\n{next_marker}"]:
                    idx = text.find(pattern, start_idx)
                    if idx != -1:
                        end_idx = min(end_idx, idx)
            
            result[key] = text[start_idx:end_idx].strip()
    
    return result


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_uids_from_file(uid_file: str, uid_key: str = "test_uids") -> set:
    """Load UIDs from a JSON file."""
    if not os.path.exists(uid_file):
        print(f"Warning: UID file not found: {uid_file}")
        return set()
    
    with open(uid_file, 'r') as f:
        data = json.load(f)
    
    uids = data.get(uid_key, [])
    return set(uids)


def load_samples_from_improved_data(
    data_dir: str,
    uid_filter: Optional[set] = None
) -> List[Dict]:
    """
    Load samples from improved_*.json files.
    
    Args:
        data_dir: Directory containing improved_*.json files
        uid_filter: Optional set of UIDs to filter (only include these)
        
    Returns:
        List of sample dictionaries
    """
    samples = []
    
    # Find all improved_*.json files
    data_files = sorted(glob.glob(os.path.join(data_dir, "improved_*.json")))
    
    print(f"Found {len(data_files)} improved_data files in {data_dir}")
    
    for filepath in data_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both 'results' and 'samples' formats
        items = data.get("results", data.get("samples", []))
        
        for result in items:
            uid = result.get("uid")
            status = result.get("status")
            
            # For 'samples' format, status might not exist - assume all are valid
            # For 'results' format, only include successful samples
            if status and status not in ['accepted', 'fixed', 'regenerated']:
                continue
            
            # Filter by UID if specified
            if uid_filter and uid not in uid_filter:
                continue
            
            modified_prompt = result.get("final_modified_prompt")
            if not modified_prompt:
                continue
            
            samples.append({
                "uid": uid,
                "modified_prompt": modified_prompt,
                "ground_truth_code": result.get("ground_truth_code"),
                "chamfer_distance": result.get("final_chamfer_distance"),
                "status": status
            })
    
    print(f"Loaded {len(samples)} samples")
    return samples


def load_samples_with_cd_filter(
    data_dir: str = DEFAULT_BATCH_DATA_DIR,
    cd_threshold: float = DEFAULT_CD_THRESHOLD,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Load samples from batch_label JSON files filtered by Chamfer Distance.
    
    Args:
        data_dir: Directory containing batch_label_*.json files
        cd_threshold: Maximum chamfer distance threshold
        max_samples: Maximum number of samples to return (None for all)
        seed: Random seed for sampling
        
    Returns:
        List of sample dictionaries with CD < threshold
    """
    samples = []
    
    # Find all batch files
    batch_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith("batch_label_") and f.endswith(".json")
    ])
    
    print(f"Found {len(batch_files)} batch files in {data_dir}")
    
    for filename in batch_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for result in data.get("results", []):
            # Skip if no modified_prompt or skipped
            if not result.get("modified_prompt") or result.get("skipped"):
                continue
            
            # Check chamfer distance from modified prompt code
            eval_data = result.get("evaluation", {})
            from_modified = eval_data.get("from_modified_prompt", {})
            
            if not from_modified.get("success"):
                continue
                
            cd = from_modified.get("chamfer_distance")
            if cd is None or cd >= cd_threshold:
                continue
            
            samples.append({
                "uid": result.get("uid"),
                "modified_prompt": result.get("modified_prompt"),
                "ground_truth_code": result.get("ground_truth_code"),
                "chamfer_distance": cd,
                "evaluation": result.get("evaluation")
            })
    
    print(f"Loaded {len(samples)} samples with CD < {cd_threshold}")
    
    if max_samples and len(samples) > max_samples:
        random.seed(seed)
        random.shuffle(samples)
        samples = samples[:max_samples]
        print(f"Randomly selected {max_samples} samples")
    
    return samples


# =============================================================================
# Async Processing Functions
# =============================================================================

def build_misleading_prompt_message(right_prompt: str, ambiguity_types: List[str], k: int) -> List[Dict]:
    """Build the message for generating a misleading prompt with few-shot examples."""
    ambiguity_description = format_ambiguity_types_for_prompt(ambiguity_types)
    # Pick the matching fewshot list based on ambiguity type
    fewshot_list = []
    for cfg in TYPE_CONFIG.values():
        if cfg["ambiguity_type"] in ambiguity_types:
            fewshot_list = cfg["fewshot"]
            break
    fewshot_examples = format_fewshot_examples(fewshot_list)
    
    # Build system prompt with few-shot examples
    system_with_examples = f"""{Misleading_system_prompt}

=== FEW-SHOT EXAMPLES ===
Below are examples showing the expected input/output format:

{fewshot_examples}
=== END OF EXAMPLES ===

Now generate a misleading description for the following input, following the same format as the examples above."""

    user_prompt = f"""RIGHT_PROMPT:
{right_prompt}

AMBIGUITY_TYPES:
{ambiguity_description}

NUM_AMBIGUITIES: {k}

Please generate the misleading description following the exact output format shown in the examples."""

    return [
        {"role": "system", "content": system_with_examples},
        {"role": "user", "content": user_prompt}
    ]


async def process_single_task(
    client: AsyncOpenAI,
    sample: Dict,
    config: Dict,
    model: str,
    idx: int,
    total: int
) -> Dict:
    """Process a single misleading prompt generation task (async)."""
    uid = sample["uid"]
    config_name = config["config_name"]
    
    print(f"  [{idx}/{total}] Processing {uid} - {config_name}...")
    
    result = {
        "uid": uid,
        "config_name": config_name,
        "ambiguity_types": config["ambiguity_types"],
        "k": config["k"],
        "original_prompt": sample["modified_prompt"],
        "original_cd": sample["chamfer_distance"],
        "ground_truth_code": sample.get("ground_truth_code"),
        "success": False,
        "misleading_description": None,
        "what_changed": None,
        "ambiguity_scan": None,
        "questions_to_ask": None,
        "answer_to_questions": None,
        "raw_response": None,
        "error": None
    }
    
    try:
        messages = build_misleading_prompt_message(
            right_prompt=sample["modified_prompt"],
            ambiguity_types=config["ambiguity_types"],
            k=config["k"]
        )
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=10000
        )
        
        raw_response = response.choices[0].message.content
        result["raw_response"] = raw_response
        
        # Parse response
        parsed = parse_misleading_response(raw_response)
        result["misleading_description"] = parsed["misleading_description"]
        result["what_changed"] = parsed["what_changed"]
        result["ambiguity_scan"] = parsed["ambiguity_scan"]
        result["questions_to_ask"] = parsed["questions_to_ask"]
        result["answer_to_questions"] = parsed["answer_to_questions"]
        result["success"] = True
        
        print(f"      ✓ {uid} - {config_name}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"      ✗ {uid} - {config_name}: {str(e)[:50]}...")
    
    return result


async def process_batch_async(
    client: AsyncOpenAI,
    tasks: List[Tuple[Dict, Dict]],  # List of (sample, config) tuples
    model: str,
    start_idx: int
) -> List[Dict]:
    """Process a batch of tasks concurrently."""
    total = len(tasks)
    
    async_tasks = []
    for i, (sample, config) in enumerate(tasks):
        task = process_single_task(
            client=client,
            sample=sample,
            config=config,
            model=model,
            idx=start_idx + i + 1,
            total=start_idx + total
        )
        async_tasks.append(task)
    
    results = await asyncio.gather(*async_tasks)
    return results


# =============================================================================
# Batch Management Functions
# =============================================================================

def get_max_batch_number(output_dir: str) -> int:
    """Find the maximum batch number from existing misleading_batch_*.json files."""
    batch_files = glob.glob(os.path.join(output_dir, 'misleading_batch_*.json'))
    max_num = 0
    for batch_file in batch_files:
        match = re.search(r'misleading_batch_(\d+)\.json', os.path.basename(batch_file))
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num


def get_processed_tasks(output_dir: str) -> set:
    """Get set of already processed (uid, config_name) tuples."""
    processed = set()
    batch_files = glob.glob(os.path.join(output_dir, 'misleading_batch_*.json'))
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            for result in batch_data.get('results', []):
                uid = result.get('uid')
                config_name = result.get('config_name')
                if uid and config_name:
                    processed.add((uid, config_name))
        except Exception as e:
            print(f"Warning: Could not load {batch_file}: {e}")
    
    return processed


def save_batch_results(
    batch_results: List[Dict],
    batch_num: int,
    output_dir: str,
    configs: List[Dict],
) -> str:
    """Save batch results to JSON file."""
    output_data = {
        'metadata': {
            'batch_number': batch_num,
            'batch_size': len(batch_results),
            'processed_at': datetime.now().isoformat(),
            'configs': [c['config_name'] for c in configs]
        },
        'results': batch_results
    }
    
    output_path = os.path.join(output_dir, f'misleading_batch_{batch_num:04d}.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Batch {batch_num} saved to {output_path}")
    return output_path


# =============================================================================
# Main Async Function
# =============================================================================

async def main_async(
    type_key: str = "underspec",
    n_samples: int = DEFAULT_N_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = DEFAULT_MODEL,
    output_dir: Optional[str] = None,
    data_dir: str = DEFAULT_BATCH_DATA_DIR,
    cd_threshold: float = DEFAULT_CD_THRESHOLD,
    seed: int = 42,
    uid_file: Optional[str] = None,
    uid_key: str = "test_uids",
    use_improved_data: bool = False,
    k_values: Optional[List[int]] = None,
):
    """
    Main async function for batch misleading prompt generation.
    
    For each sample, generates misleading prompts with K=1, 2, 3.
    
    Args:
        n_samples: Number of samples to process
        batch_size: Batch size for async processing
        model: OpenAI model name
        output_dir: Output directory
        data_dir: Input data directory
        cd_threshold: Chamfer distance threshold (for batch_label format)
        seed: Random seed
        uid_file: Optional JSON file with UID lists to filter by
        uid_key: Key in uid_file to use (e.g., "test_uids" or "train_uids")
        use_improved_data: If True, load from improved_*.json files
        k_values: List of K values to use (default: [1, 2, 3])
    """
    import time
    total_start_time = time.time()

    if type_key not in TYPE_CONFIG:
        raise ValueError(f"Unknown --type '{type_key}'. Valid: {list(TYPE_CONFIG)}")

    configs_to_use = build_batch_configs(type_key, k_values)
    if output_dir is None:
        output_dir = os.environ.get(
            "MISLEADING_OUTPUT_DIR", TYPE_CONFIG[type_key]["default_output_dir"]
        )

    print(f"\n{'='*60}")
    print(f"BATCH MISLEADING PROMPT GENERATION ({type_key})")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"N Samples: {n_samples}")
    print(f"K values: {[c['k'] for c in configs_to_use]}")
    print(f"Configs per sample: {len(configs_to_use)}")
    print(f"Total expected tasks: {n_samples * len(configs_to_use)}")
    print(f"Batch Size: {batch_size}")
    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"UID File: {uid_file}")
    print(f"UID Key: {uid_key}")
    print(f"Use Improved Data: {use_improved_data}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load already processed tasks
    processed_tasks = get_processed_tasks(output_dir)
    if processed_tasks:
        print(f"Found {len(processed_tasks)} already processed tasks")
    
    # Initialize async client
    client = AsyncOpenAI()
    
    # Load UID filter if specified
    uid_filter = None
    if uid_file:
        uid_filter = load_uids_from_file(uid_file, uid_key)
        print(f"Loaded {len(uid_filter)} UIDs from {uid_file} (key: {uid_key})")
    
    # Load samples
    if use_improved_data:
        samples = load_samples_from_improved_data(
            data_dir=data_dir,
            uid_filter=uid_filter
        )
    else:
        samples = load_samples_with_cd_filter(
            data_dir=data_dir,
            cd_threshold=cd_threshold,
            max_samples=None,
            seed=seed
        )
        # Apply UID filter if specified
        if uid_filter:
            samples = [s for s in samples if s["uid"] in uid_filter]
            print(f"After UID filter: {len(samples)} samples")
    
    # Apply max_samples limit if specified
    if n_samples and len(samples) > n_samples:
        samples = samples[:n_samples]
        print(f"Limited to first {n_samples} samples")
    
    if len(samples) == 0:
        print("No samples found with CD < threshold. Exiting.")
        await client.close()
        return
    
    # Build all tasks: (sample, config) pairs
    all_tasks = []
    for sample in samples:
        for config in configs_to_use:
            task_key = (sample["uid"], config["config_name"])
            if task_key not in processed_tasks:
                all_tasks.append((sample, config))
    
    print(f"Total tasks to process: {len(all_tasks)} (after filtering already processed)")
    
    if len(all_tasks) == 0:
        print("All tasks already processed. Nothing to do.")
        await client.close()
        return
    
    # Process in batches
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    batch_offset = get_max_batch_number(output_dir)
    
    if batch_offset > 0:
        print(f"Found existing batch files up to {batch_offset:04d}, new batches start from {batch_offset + 1:04d}")
    
    all_results = []
    tasks_processed = 0
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, len(all_tasks))
        batch_tasks = all_tasks[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} (tasks {start_idx + 1}-{end_idx})")
        print(f"{'='*60}")
        
        batch_start_time = time.time()
        
        batch_results = await process_batch_async(
            client=client,
            tasks=batch_tasks,
            model=model,
            start_idx=tasks_processed
        )
        
        batch_time = time.time() - batch_start_time
        
        # Save batch
        actual_batch_num = batch_offset + batch_num
        save_batch_results(batch_results, actual_batch_num, output_dir, configs_to_use)
        
        # Stats
        successful = sum(1 for r in batch_results if r.get("success"))
        failed = len(batch_results) - successful
        print(f"  Batch time: {batch_time:.1f}s | Success: {successful} | Failed: {failed}")
        
        all_results.extend(batch_results)
        tasks_processed += len(batch_tasks)
    
    # Final summary
    total_time = time.time() - total_start_time
    total_successful = sum(1 for r in all_results if r.get("success"))
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total tasks processed: {len(all_results)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {len(all_results) - total_successful}")
    print(f"Total time: {total_time:.1f}s")
    if len(all_results) > 0:
        print(f"Average time per task: {total_time/len(all_results):.2f}s")
    print(f"{'='*60}")
    
    await client.close()
    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch generate misleading CAD descriptions (underspec or conflict)"
    )
    parser.add_argument("--type", type=str, choices=list(TYPE_CONFIG), default="underspec",
                        help="Ambiguity family to generate: 'underspec' or 'conflict' (default: underspec)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max number of samples to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for async processing (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: per-type default from TYPE_CONFIG)")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("IMPROVED_DATA_DIR", "./sft/improved_data"),
                        help="Input data directory (default: improved_data)")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test",
                        help="Use 'train' or 'test' split from train_val_test.json (default: test)")
    parser.add_argument("--train-uid-set", type=str,
                        choices=["train_uids", "train_type2_uids"], default="train_uids",
                        help="For train split: 'train_uids' (original) or 'train_type2_uids' (additional). "
                             "Only applies when --split train (default: train_uids)")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="K values to use (e.g., --k 1 2 3). Default: all [1, 2, 3]")

    args = parser.parse_args()

    if args.split == "test" and args.train_uid_set != "train_uids":
        parser.error("--train-uid-set can only be used with --split train")

    # Map split to uid_key
    uid_key = "test_uids" if args.split == "test" else args.train_uid_set

    asyncio.run(main_async(
        type_key=args.type,
        n_samples=args.num_samples,
        batch_size=args.batch_size,
        model=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        cd_threshold=DEFAULT_CD_THRESHOLD,
        seed=42,
        uid_file=DEFAULT_UID_FILE,
        uid_key=uid_key,
        use_improved_data=True,
        k_values=args.k
    ))


if __name__ == "__main__":
    main()

