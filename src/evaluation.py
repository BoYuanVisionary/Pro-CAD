"""
Evaluation metrics for CAD Agent.
Functions for computing Chamfer Distance and other metrics.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Optional, Tuple
import trimesh

from .mesh_utils import (
    mesh_to_pointcloud,
    normalize_mesh,
    load_mesh_from_stl,
    cadquery_to_mesh
)


def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.
    
    CD = (1/|P1|) * sum(min_q d(p,q)^2) + (1/|P2|) * sum(min_p d(q,p)^2)
    
    Args:
        pts1: First point cloud, shape (N, 3)
        pts2: Second point cloud, shape (M, 3)
        
    Returns:
        Chamfer distance (mean of squared distances)
    """
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    
    return (np.mean(d1**2) + np.mean(d2**2)) / 2


def compute_baseline_cd(
    mesh: trimesh.Trimesh,
    n_points: int = 8192,
    seed1: int = 42,
    seed2: int = 123
) -> float:
    """
    Compute baseline Chamfer Distance (lower bound) by sampling mesh twice.
    
    This represents the inherent sampling noise - the best possible CD 
    we can achieve when comparing two point clouds from the same mesh.
    
    Args:
        mesh: Normalized trimesh object
        n_points: Number of points to sample
        seed1: Random seed for first sampling
        seed2: Random seed for second sampling
        
    Returns:
        Baseline chamfer distance
    """
    np.random.seed(seed1)
    pts1 = mesh_to_pointcloud(mesh, n_points=n_points)
    
    np.random.seed(seed2)
    pts2 = mesh_to_pointcloud(mesh, n_points=n_points)
    
    return chamfer_distance(pts1, pts2)


def evaluate_sample(
    gt_mesh: trimesh.Trimesh,
    gen_mesh: trimesh.Trimesh,
    n_points: int = 8192,
    compute_baseline: bool = True
) -> Dict:
    """
    Evaluate a generated mesh against ground truth.
    
    Args:
        gt_mesh: Ground truth mesh (will be normalized)
        gen_mesh: Generated mesh (will be normalized)
        n_points: Number of points for point cloud sampling
        compute_baseline: Whether to compute baseline CD
        
    Returns:
        Dictionary with evaluation results:
            - cd: Chamfer distance
            - baseline_cd: Baseline CD (if compute_baseline=True)
            - ratio: CD / baseline_cd ratio
            - quality: Quality assessment string
    """
    # Normalize meshes
    gt_mesh_norm = normalize_mesh(gt_mesh)
    gen_mesh_norm = normalize_mesh(gen_mesh)
    
    # Sample point clouds
    gt_pts = mesh_to_pointcloud(gt_mesh_norm, n_points=n_points)
    gen_pts = mesh_to_pointcloud(gen_mesh_norm, n_points=n_points)
    
    # Compute CD
    cd = chamfer_distance(gt_pts, gen_pts)
    
    result = {
        'cd': cd,
        'gt_vertices': len(gt_mesh_norm.vertices),
        'gt_faces': len(gt_mesh_norm.faces),
        'gen_vertices': len(gen_mesh_norm.vertices),
        'gen_faces': len(gen_mesh_norm.faces),
    }
    
    # Compute baseline if requested
    if compute_baseline:
        baseline_cd = compute_baseline_cd(gt_mesh_norm, n_points=n_points)
        result['baseline_cd'] = baseline_cd
        result['ratio'] = cd / baseline_cd if baseline_cd > 0 else float('inf')
    
    # Quality assessment
    if cd < 0.0001:
        result['quality'] = "EXCELLENT"
    elif cd < 0.0005:
        result['quality'] = "GOOD"
    elif cd < 0.001:
        result['quality'] = "MODERATE"
    else:
        result['quality'] = "POOR"
    
    return result


def evaluate_from_paths(
    gt_stl_path: str,
    gen_code: str,
    n_points: int = 8192
) -> Dict:
    """
    Evaluate generated code against ground truth STL.
    
    Args:
        gt_stl_path: Path to ground truth STL file
        gen_code: Generated CadQuery code
        n_points: Number of points for sampling
        
    Returns:
        Evaluation results dictionary
    """
    gt_mesh = load_mesh_from_stl(gt_stl_path)
    gen_mesh = cadquery_to_mesh(gen_code)
    
    return evaluate_sample(gt_mesh, gen_mesh, n_points=n_points)


def is_outlier(result: Dict, threshold_ratio: float = 5.0) -> bool:
    """
    Check if a sample is an outlier based on CD/baseline ratio.
    
    Args:
        result: Evaluation result dictionary
        threshold_ratio: Threshold for outlier detection
        
    Returns:
        True if sample is an outlier
    """
    ratio = result.get('ratio', 0)
    return ratio > threshold_ratio


def compute_statistics(results: list) -> Dict:
    """
    Compute statistics over multiple evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Statistics dictionary
    """
    cds = [r['cd'] for r in results if 'cd' in r]
    
    stats = {
        'count': len(cds),
        'mean_cd': np.mean(cds) if cds else None,
        'median_cd': np.median(cds) if cds else None,
        'min_cd': np.min(cds) if cds else None,
        'max_cd': np.max(cds) if cds else None,
        'std_cd': np.std(cds) if cds else None,
    }
    
    # Baseline stats if available
    baselines = [r['baseline_cd'] for r in results if 'baseline_cd' in r]
    if baselines:
        stats['mean_baseline'] = np.mean(baselines)
    
    # Ratio stats if available
    ratios = [r['ratio'] for r in results if 'ratio' in r and r['ratio'] != float('inf')]
    if ratios:
        stats['mean_ratio'] = np.mean(ratios)
        stats['median_ratio'] = np.median(ratios)
        stats['max_ratio'] = np.max(ratios)
    
    # Quality distribution
    qualities = [r.get('quality', 'UNKNOWN') for r in results]
    stats['quality_distribution'] = {
        q: qualities.count(q) for q in set(qualities)
    }
    
    return stats


def extract_code(response: str) -> str:
    """
    Extract Python code from model response.
    
    Args:
        response: Raw model response (may contain markdown code blocks)
        
    Returns:
        Extracted code string
    """
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def calculate_question_score(judge_json, ground_truth_questions):
    """
    judge_json: The dictionary output from the LLM Judge
    ground_truth_questions: The original list of expected questions
    """
    
    # 1. Count the metrics
    tp = len(judge_json.get("matched_questions", []))
    fp = len(judge_json.get("hallucinated_questions", []))
    
    total_needed = len(ground_truth_questions)
    fn = total_needed - tp
    
    # 2. Safety checks for division by zero
    if total_needed == 0:
        return 1.0 if fp == 0 else 0.0 # If no questions needed, success is asking nothing
        
    # 3. Calculate Scores
    recall = tp / total_needed
    
    total_generated = tp + fp
    precision = 1.0 if total_generated == 0 else (tp / total_generated)
    
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "score": f1_score,       # Use this as your main metric (0 to 1)
        "recall": recall,        # Use to debug if model is "lazy" (misses things)
        "precision": precision,  # Use to debug if model is "chatty" (hallucinates)
        "details": {
            "found": tp,
            "missed": fn,
            "extra": fp
        }
    }
