"""
Batch processing script for generating LLM labels and evaluating CadQuery code.
Processes samples in batches using async OpenAI API.
"""

import pickle
import random
import os
import sys
import json
import traceback
import asyncio
import glob
import numpy as np
import trimesh
import cadquery as cq
from datetime import datetime
from scipy.spatial import cKDTree
from openai import AsyncOpenAI

# Add tools directory to path (set TOOLS_DIR or adjust path to your repo)
TOOLS_DIR = os.environ.get("TOOLS_DIR", ".")
sys.path.append(TOOLS_DIR)
from render_image import render_image

from config.code_generation import (
    DESCRIPTION_SYSTEM_PROMPT,
    CODE_GENERATION_SYSTEM_PROMPT
)


# =============================================================================
# Configuration
# =============================================================================

# Data paths - use filter_results.json as source (set DATA_ROOT env var or use relative paths)
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
FILTER_RESULTS_PATH = os.path.join(DATA_ROOT, 'text2cad/filter_results.json')
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'text2cad/train.pkl')
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'text2cad/test.pkl')
VAL_DATA_PATH = os.path.join(DATA_ROOT, 'text2cad/val.pkl')
GT_CODE_DIR = os.path.join(DATA_ROOT, 'text2cad/cadquery')
GT_MESH_DIR = os.path.join(DATA_ROOT, 'text2cad/deepcad_mesh')

DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_N_SAMPLES = 10000
DEFAULT_BATCH_SIZE = 32
DEFAULT_OUTPUT_DIR = 'sft/filtered_data'

# Set OPENAI_API_KEY in your environment before running (e.g. export OPENAI_API_KEY=sk-...)


# =============================================================================
# Evaluation Functions
# =============================================================================

def cadquery_to_mesh(code: str) -> trimesh.Trimesh:
    """Execute CadQuery code and convert result to trimesh."""
    namespace = {'cq': cq}
    exec(code, namespace)
    
    if 'r' not in namespace:
        raise ValueError("Code must define variable 'r' with CadQuery result")
    
    compound = namespace['r'].val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    
    mesh = trimesh.Trimesh(
        vertices=[(v.x, v.y, v.z) for v in vertices],
        faces=faces
    )
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to unit cube centered at (0.5, 0.5, 0.5)."""
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_translation([0.5, 0.5, 0.5])
    return mesh


def chamfer_distance(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, n_points: int = 8192) -> float:
    """Compute Chamfer Distance between two meshes."""
    pts1 = mesh1.sample(n_points)
    pts2 = mesh2.sample(n_points)
    
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    
    return (np.mean(d1**2) + np.mean(d2**2)) / 2


def evaluate_code(code: str):
    """Evaluate a code string and return results with error info."""
    result = {
        'success': False,
        'chamfer_distance': None,
        'error': None,
        'error_traceback': None,
        'mesh_vertices': None,
        'mesh_faces': None
    }
    
    if code is None:
        result['error'] = "Code is None"
        return result, None
    
    try:
        gen_mesh = cadquery_to_mesh(code)
        gen_mesh = normalize_mesh(gen_mesh)
        result['success'] = True
        result['mesh_vertices'] = len(gen_mesh.vertices)
        result['mesh_faces'] = len(gen_mesh.faces)
        return result, gen_mesh
    except Exception as e:
        result['error'] = str(e)
        result['error_traceback'] = traceback.format_exc()
        return result, None


def evaluate_sample(result, uid):
    """Evaluate a processed sample using Chamfer Distance."""
    eval_results = {
        'uid': uid,
        'gt_mesh_success': False,
        'from_modified_prompt': {
            'code': result['generated_code_from_modified_prompt'],
            'success': False,
            'chamfer_distance': None,
            'error': None,
            'error_traceback': None,
            'mesh_vertices': None,
            'mesh_faces': None
        }
    }
    
    if result.get('processing_error'):
        eval_results['error'] = 'Skipped due to processing error'
        return eval_results
    
    # Load ground truth mesh
    gt_mesh_path = os.path.join(GT_MESH_DIR, f"{uid}.stl")
    gt_mesh = None
    try:
        gt_mesh = trimesh.load(gt_mesh_path)
        gt_mesh = normalize_mesh(gt_mesh)
        eval_results['gt_mesh_success'] = True
    except Exception as e:
        return eval_results
    
    # Evaluate code from modified prompt
    if result['generated_code_from_modified_prompt']:
        result_modified, gen_mesh_modified = evaluate_code(result['generated_code_from_modified_prompt'])
        eval_results['from_modified_prompt'].update(result_modified)
        
        if gen_mesh_modified is not None and gt_mesh is not None:
            cd = chamfer_distance(gen_mesh_modified, gt_mesh)
            eval_results['from_modified_prompt']['chamfer_distance'] = cd
    
    return eval_results


# =============================================================================
# Processing Functions
# =============================================================================

def evaluate_gt_code(gt_code: str, uid: str, cd_threshold: float = 0.0005):
    """
    Evaluate ground truth code against GT mesh.
    Returns (success, chamfer_distance, error_msg)
    """
    try:
        # Load ground truth mesh
        gt_mesh_path = os.path.join(GT_MESH_DIR, f"{uid}.stl")
        gt_mesh = trimesh.load(gt_mesh_path)
        gt_mesh = normalize_mesh(gt_mesh)
        
        # Execute GT code and get mesh
        gen_mesh = cadquery_to_mesh(gt_code)
        gen_mesh = normalize_mesh(gen_mesh)
        
        # Compute chamfer distance
        cd = chamfer_distance(gen_mesh, gt_mesh)
        
        return True, cd, None
    except Exception as e:
        return False, None, str(e)


async def process_sample(client, sample_data, model, idx, total, cd_threshold=0.0005):
    """Process a single sample through the full pipeline (async)."""
    uid = sample_data['uid']
    original_prompt = sample_data.get('original_prompt', '')
    gt_code = sample_data['gt_code']
    gt_code_path = sample_data['gt_code_path']
    
    print(f"  [{idx}/{total}] Processing {uid}...")
    
    result = {
        'uid': uid,
        'original_prompt': original_prompt,  # Saved for reference only, NOT used in LLM prompts
        'modified_prompt': None,
        'generated_code_from_modified_prompt': None,
        'ground_truth_code': gt_code,
        'gt_code_path': gt_code_path,
        'gt_code_cd': None,
        'evaluation': None,
        'processing_error': None,
        'skipped': False
    }
    
    try:
        # Step 0: Evaluate ground truth code first
        gt_success, gt_cd, gt_error = evaluate_gt_code(gt_code, uid, cd_threshold)
        result['gt_code_cd'] = gt_cd
        
        if not gt_success:
            result['processing_error'] = f"GT code evaluation failed: {gt_error}"
            result['skipped'] = True
            print(f"      ✗ GT code failed: {gt_error[:50]}...")
            return result
        
        if gt_cd >= cd_threshold:
            result['processing_error'] = f"GT code CD ({gt_cd:.6f}) >= threshold ({cd_threshold})"
            result['skipped'] = True
            print(f"      ✗ Skipped: GT CD={gt_cd:.6f} >= {cd_threshold}")
            return result
        
        print(f"      GT CD={gt_cd:.6f} ✓")
        
        # Step 1: Render image from ground truth code
        render_result = render_image(gt_code, img_size=256, num_views=4, save=False)
        
        if not render_result['success']:
            result['processing_error'] = f"Rendering failed: {render_result['image']}"
            print(f"      ✗ Rendering failed")
            return result
        
        image_base64 = render_result['image']
        
        # Step 2: Generate description from image + code
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_SYSTEM_PROMPT},
                {"type": "text", "text": f"\n\n=== GROUND TRUTH CADQUERY CODE ===\n\n```python\n{gt_code}\n```"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}}
            ]
        }]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=10000
        )
        
        gpt_description = response.choices[0].message.content
        result['modified_prompt'] = gpt_description
        
        # Step 3: Generate code from modified prompt
        user_prompt = f"""Generate CadQuery Python code for the following 3D shape:

{gpt_description}

Output only the Python code, no explanations."""
        
        messages = [
            {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=10000
        )
        
        generated_code = response.choices[0].message.content
        
        # Extract code if wrapped in markdown
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        result['generated_code_from_modified_prompt'] = generated_code
        
        # Step 4: Evaluate
        eval_results = evaluate_sample(result, uid)
        result['evaluation'] = eval_results
        
        # Print result
        cd_modified = eval_results['from_modified_prompt'].get('chamfer_distance')
        if cd_modified is not None:
            print(f"      ✓ CD (modified): {cd_modified:.6f}")
        else:
            print(f"      ✗ Evaluation failed")
        
    except Exception as e:
        result['processing_error'] = f"Processing failed: {str(e)}"
        result['error_traceback'] = traceback.format_exc()
        print(f"      ✗ Error: {str(e)[:50]}...")
    
    return result


async def process_batch(client, batch_samples, model, start_idx, total, cd_threshold=0.0005):
    """Process a batch of samples concurrently."""
    tasks = []
    for i, sample_data in enumerate(batch_samples):
        idx = start_idx + i + 1
        task = process_sample(client, sample_data, model, idx, total, cd_threshold)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results


def get_max_batch_number(output_dir='.'):
    """Find the maximum batch number from existing batch_label_*.json files."""
    import re
    batch_files = glob.glob(os.path.join(output_dir, 'batch_label_*.json'))
    max_num = 0
    for batch_file in batch_files:
        # Extract number from filename like batch_label_0087.json
        match = re.search(r'batch_label_(\d+)\.json', os.path.basename(batch_file))
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num


def save_batch_results(batch_results, batch_num, output_dir='.', batch_offset=0):
    """Save batch results to JSON file."""
    for result in batch_results:
        if '_sample_data' in result:
            del result['_sample_data']
    
    actual_batch_num = batch_offset + batch_num
    
    output_data = {
        'metadata': {
            'batch_number': actual_batch_num,
            'batch_size': len(batch_results),
            'processed_at': datetime.now().isoformat()
        },
        'results': batch_results
    }
    
    output_path = os.path.join(output_dir, f'batch_label_{actual_batch_num:04d}.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Batch {batch_num} saved to {output_path}")
    return output_path


# =============================================================================
# Main Function
# =============================================================================

async def main_async(n_samples, batch_size, model, output_dir, seed=42, cd_threshold=0.0005):
    """Main async function for batch processing."""
    import time
    total_start_time = time.time()
    
    print(f"\n{'='*60}")
    print("BATCH LABEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"N Samples: {n_samples}")
    print(f"Batch Size: {batch_size}")
    print(f"CD Threshold: {cd_threshold}")
    print(f"Output Dir: {output_dir}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load already processed UIDs from output directory
    processed_uids = set()
    batch_files = glob.glob(os.path.join(output_dir, 'batch_label_*.json'))
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            for result in batch_data.get('results', []):
                if 'uid' in result:
                    processed_uids.add(result['uid'])
        except Exception as e:
            print(f"Warning: Could not load {batch_file}: {e}")
    
    if processed_uids:
        print(f"Found {len(processed_uids)} already processed samples in {output_dir}")
    
    # Initialize async client
    client = AsyncOpenAI()
    
    # Load filter_results.json to get valid UIDs
    print("Loading filter_results.json...")
    with open(FILTER_RESULTS_PATH, 'r') as f:
        filter_results = json.load(f)
    valid_uids = set(filter_results['keep'])
    print(f"Valid UIDs from filter_results.json: {len(valid_uids)}")
    
    # Load train, test, val data to get original descriptions (for reference only, not used in LLM prompts)
    print("Loading train/test/val data for original descriptions...")
    uid_to_description = {}
    for pkl_path in [TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH]:
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            for s in data:
                uid_to_description[s['uid']] = s.get('description', '')
    print(f"Loaded descriptions for {len(uid_to_description)} UIDs")
    
    # Find UIDs that have both GT code and mesh files
    samples_with_gt = []
    for uid in valid_uids:
        cadquery_path = os.path.join(GT_CODE_DIR, f"{uid}.py")
        gt_mesh_path = os.path.join(GT_MESH_DIR, f"{uid}.stl")
        if os.path.exists(cadquery_path) and os.path.exists(gt_mesh_path):
            samples_with_gt.append({
                'uid': uid,
                'original_prompt': uid_to_description.get(uid, '')
            })
    print(f"Samples with GT code and mesh: {len(samples_with_gt)}")
    
    # Randomly select n samples
    random.seed(seed)
    random.shuffle(samples_with_gt)
    selected_samples = samples_with_gt[:n_samples]
    print(f"Selected {len(selected_samples)} samples for processing")
    
    # Filter out already processed samples
    selected_samples = [s for s in selected_samples if s['uid'] not in processed_uids]
    print(f"After filtering already processed: {len(selected_samples)} samples remaining")
    
    if len(selected_samples) == 0:
        print("All samples already processed. Nothing to do.")
        await client.close()
        return
    
    # Prepare samples with GT code loaded
    samples_to_process = []
    for s in selected_samples:
        uid = s['uid']
        gt_code_path = os.path.join(GT_CODE_DIR, f"{uid}.py")
        with open(gt_code_path, 'r') as f:
            gt_code = f.read()
        samples_to_process.append({
            'uid': uid,
            'original_prompt': s.get('original_prompt', ''),
            'gt_code': gt_code,
            'gt_code_path': gt_code_path
        })
    
    # Process in batches
    all_results = []
    total_batches = (len(samples_to_process) + batch_size - 1) // batch_size
    
    # Get the max existing batch number to avoid overwriting
    batch_offset = get_max_batch_number(output_dir)
    if batch_offset > 0:
        print(f"Found existing batch files up to {batch_offset:04d}, new batches will start from {batch_offset + 1:04d}")
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, len(samples_to_process))
        batch_samples = samples_to_process[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} (samples {start_idx + 1}-{end_idx})")
        print(f"{'='*60}")
        
        batch_start_time = time.time()
        batch_results = await process_batch(
            client, batch_samples, model, start_idx, len(samples_to_process), cd_threshold
        )
        batch_elapsed = time.time() - batch_start_time
        print(f"\n  Batch {batch_num} completed in {batch_elapsed:.2f}s ({batch_elapsed/len(batch_samples):.2f}s per sample)")
        
        save_batch_results(batch_results, batch_num, output_dir, batch_offset)
        all_results.extend(batch_results)
    
    # Close client
    await client.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    skipped = sum(1 for r in all_results if r.get('skipped', False))
    processed = len(all_results) - skipped
    successful = sum(1 for r in all_results 
                     if r.get('evaluation', {}).get('from_modified_prompt', {}).get('success', False))
    
    print(f"Total samples: {len(all_results)}")
    print(f"Skipped (GT CD >= {cd_threshold}): {skipped}")
    print(f"Processed: {processed}")
    print(f"Successful: {successful}")
    
    cd_list = [r.get('evaluation', {}).get('from_modified_prompt', {}).get('chamfer_distance') 
               for r in all_results 
               if r.get('evaluation', {}).get('from_modified_prompt', {}).get('chamfer_distance') is not None]
    
    if cd_list:
        print(f"\nChamfer Distance: mean={np.mean(cd_list):.6f}, min={np.min(cd_list):.6f}, max={np.max(cd_list):.6f}")
    
    # Print timing info
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"Average time per sample: {total_elapsed/len(all_results):.2f}s")
    
    # Save final summary
    output_path = os.path.join(output_dir, 'batch_label_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_samples': len(all_results),
                'model': model,
                'processed_at': datetime.now().isoformat(),
                'total_time_seconds': total_elapsed,
                'avg_time_per_sample': total_elapsed / len(all_results) if all_results else 0
            },
            'summary': {
                'total': len(all_results),
                'skipped': skipped,
                'processed': processed,
                'successful': successful,
                'cd_threshold': cd_threshold,
                'avg_cd': float(np.mean(cd_list)) if cd_list else None,
                'min_cd': float(np.min(cd_list)) if cd_list else None,
                'max_cd': float(np.max(cd_list)) if cd_list else None
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\nFinal results saved to {output_path}")


def main():
    """Entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch processing for CAD label generation')
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES,
                        help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of samples per batch')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Model name for inference')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    parser.add_argument('--cd_threshold', type=float, default=0.0002,
                        help='Chamfer distance threshold for GT code filtering')
    
    args = parser.parse_args()
    
    asyncio.run(main_async(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        cd_threshold=args.cd_threshold
    ))


if __name__ == "__main__":
    main()
