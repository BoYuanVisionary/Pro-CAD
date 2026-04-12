"""
Preprocessing utilities for CAD Agent.
Generate mesh files from DeepCAD JSON using UIDs from cadquery folder.
"""

import os
import sys
import json
import numpy as np
import trimesh
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from tqdm import tqdm

# Add DeepCAD path for imports (set DEEPCAD_PATH env var to your DeepCAD repo)
DEEPCAD_PATH = os.environ.get("DEEPCAD_PATH", "./data/DeepCAD")
sys.path.append(DEEPCAD_PATH)
from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import logging
import warnings

logging.getLogger("trimesh").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r".*face_.*(normals|models).*(triangles|faces).*",
    category=UserWarning,
)


# Paths (set DATA_ROOT env var or use relative paths)
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
CADQUERY_DIR = os.path.join(DATA_ROOT, "text2cad/cadquery")
JSON_DIR = os.path.join(DATA_ROOT, "DeepCAD/dataset/data/cad_json")
MESH_OUTPUT_DIR = os.path.join(DATA_ROOT, "text2cad/deepcad_mesh")

# Tessellation parameters
LINEAR_DEFLECTION = 0.0001
ANGULAR_DEFLECTION = 0.1

# Number of parallel workers
N_JOBS = 10


def get_uids_from_cadquery(cadquery_dir: str = CADQUERY_DIR) -> list:
    """
    Get list of UIDs from cadquery folder.
    
    Args:
        cadquery_dir: Directory containing .py files
        
    Returns:
        List of UIDs (file names without extension)
    """
    uids = []
    for f in os.listdir(cadquery_dir):
        if f.endswith('.py'):
            uids.append(f.replace('.py', ''))
    return sorted(uids)


def uid_to_json_path(uid: str, json_dir: str = JSON_DIR) -> str:
    """
    Convert UID to JSON file path.
    UIDs like '00123456' map to '0012/00123456.json'
    """
    folder = uid[:4]
    return os.path.join(json_dir, folder, f"{uid}.json")


def process_one(uid: str, json_dir: str = JSON_DIR, output_dir: str = MESH_OUTPUT_DIR) -> dict:
    """
    Process a single UID: load JSON, create CAD, save mesh.
    
    Args:
        uid: Sample UID
        json_dir: Directory containing JSON files
        output_dir: Directory to save STL files
        
    Returns:
        Result dict with status
    """
    output_path = os.path.join(output_dir, f"{uid}.stl")
    
    # Skip if already exists
    if os.path.exists(output_path):
        return {'uid': uid, 'status': 'skipped'}
    
    # Find JSON file
    json_path = uid_to_json_path(uid, json_dir)
    if not os.path.exists(json_path):
        return {'uid': uid, 'status': 'failed', 'error': 'JSON not found'}
    
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create CAD (same as json2pc.py)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
        
        # Tessellate with fine parameters
        mesh = BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION, False, ANGULAR_DEFLECTION)
        mesh.Perform()
        
        # Save as binary STL
        write_stl_file(shape, output_path, mode="binary")
        
        return {'uid': uid, 'status': 'success'}
        
    except Exception as e:
        return {'uid': uid, 'status': 'failed', 'error': str(e)}


def generate_all_meshes(
    cadquery_dir: str = CADQUERY_DIR,
    json_dir: str = JSON_DIR,
    output_dir: str = MESH_OUTPUT_DIR,
    n_jobs: int = N_JOBS
):
    """
    Generate mesh files for all UIDs in cadquery folder.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get UIDs from cadquery folder
    uids = get_uids_from_cadquery(cadquery_dir)
    print(f"Found {len(uids)} UIDs in {cadquery_dir}")
    
    # Process in parallel
    print(f"Processing with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_one)(uid, json_dir, output_dir) for uid in uids
    )
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total UIDs: {len(uids)}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    return results


# =============================================================================
# Filtering Functions - Remove Similar Objects
# =============================================================================

CD_THRESHOLD = 0.0005  # Chamfer distance threshold for duplicates
AREA_PRECISION = 4     # Decimal places for area/volume grouping
VOLUME_PRECISION = 4


def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Compute Chamfer Distance between two point clouds."""
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    return (np.mean(d1**2) + np.mean(d2**2)) / 2


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to unit cube."""
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_translation([0.5, 0.5, 0.5])
    return mesh


def get_mesh_fingerprint(mesh: trimesh.Trimesh, area_prec: int = AREA_PRECISION, vol_prec: int = VOLUME_PRECISION) -> tuple:
    """Get fingerprint based on area and volume for fast grouping."""
    m = normalize_mesh(mesh)
    return (round(m.area, area_prec), round(m.volume, vol_prec))


def compute_fingerprint_for_file(filepath: str) -> dict:
    """Compute fingerprint for a single mesh file."""
    uid = os.path.basename(filepath).replace('.stl', '')
    try:
        mesh = trimesh.load(filepath)
        fp = get_mesh_fingerprint(mesh)
        return {'uid': uid, 'path': filepath, 'fingerprint': fp, 'status': 'success'}
    except Exception as e:
        return {'uid': uid, 'path': filepath, 'fingerprint': None, 'status': 'failed', 'error': str(e)}


def filter_similar_objects(
    mesh_dir: str,
    cd_threshold: float = CD_THRESHOLD,
    n_points: int = 4096,
    n_jobs: int = N_JOBS,
    max_samples: int = None
) -> dict:
    """
    Filter out similar objects based on area, volume, and Chamfer Distance.
    
    Args:
        mesh_dir: Directory containing STL mesh files
        cd_threshold: Chamfer distance threshold (default 0.0001)
        n_points: Number of points for CD computation
        n_jobs: Number of parallel workers
        max_samples: Maximum number of samples to process (None = all)
        
    Returns:
        Dict with 'keep' (UIDs to keep) and 'remove' (UIDs to remove)
    """
    # Get all mesh files
    mesh_files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith('.stl')]
    print(f"Found {len(mesh_files)} mesh files")
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(mesh_files):
        import random
        random.seed(42)
        mesh_files = random.sample(mesh_files, max_samples)
        print(f"Using {max_samples} samples (randomly selected)")
    
    # Step 1: Compute fingerprints in parallel
    print("Step 1: Computing fingerprints (area, volume)...")
    fingerprints = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_fingerprint_for_file)(f) for f in mesh_files
    )
    
    # Group by fingerprint
    groups = defaultdict(list)
    for fp_result in fingerprints:
        if fp_result['status'] == 'success' and fp_result['fingerprint'] is not None:
            groups[fp_result['fingerprint']].append(fp_result)
    
    print(f"Grouped into {len(groups)} unique fingerprints")
    
    # Count potential duplicates and show original group size distribution
    potential_dups = sum(1 for g in groups.values() if len(g) > 1)
    print(f"Groups with potential duplicates: {potential_dups}")
    
    # Original group sizes before filtering
    orig_sizes = Counter(len(g) for g in groups.values())
    print("Original group size distribution:")
    for size in sorted(orig_sizes.keys())[:10]:  # Show top 10 sizes
        print(f"  Size {size}: {orig_sizes[size]} groups")
    if len(orig_sizes) > 10:
        print(f"  ... and {len(orig_sizes) - 10} more size categories")
    
    # Step 2: Within each group, use CD + Union-Find for transitive grouping (PARALLEL)
    print("Step 2: Computing Chamfer Distance within groups (transitive clustering)...")
    
    # Process a single group - returns (keep_list, remove_list)
    def process_group(group, cd_threshold, n_points):
        import random
        random.seed(42)
        
        # Union-Find helper functions
        def find(parent, i):
            if parent[i] != i:
                parent[i] = find(parent, parent[i])
            return parent[i]
        
        def union(parent, rank, x, y):
            xroot = find(parent, x)
            yroot = find(parent, y)
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1
        
        keep_list = []
        remove_list = []
        error_keep = []
        
        if len(group) == 1:
            return [group[0]['uid']], [], [1], []  # Single item = cluster of size 1, no cluster info
        
        # Load all meshes, compute point clouds, and build KDTrees ONCE
        items_data = []  # List of (uid, pts, tree)
        for item in group:
            try:
                mesh = trimesh.load(item['path'], force='mesh', process=False)
                mesh_norm = normalize_mesh(mesh)
                np.random.seed(42)
                pts = mesh_norm.sample(n_points)
                tree = cKDTree(pts)  # Build tree once per item
                items_data.append((item['uid'], pts, tree))
            except:
                error_keep.append(item['uid'])
        
        if len(items_data) <= 1:
            # Each remaining item is its own cluster of size 1
            cluster_sizes = [1] * (len(items_data) + len(error_keep))
            return [uid for uid, _, _ in items_data] + error_keep, [], cluster_sizes, []
        
        # Initialize Union-Find
        n = len(items_data)
        parent = list(range(n))
        rank = [0] * n
        cd_matrix = {}
        
        # Compute pairwise CD using pre-built trees
        for i in range(n):
            for j in range(i + 1, n):
                uid_i, pts_i, tree_i = items_data[i]
                uid_j, pts_j, tree_j = items_data[j]
                d1, _ = tree_i.query(pts_j)  # Reuse pre-built tree
                d2, _ = tree_j.query(pts_i)  # Reuse pre-built tree
                cd = (np.mean(d1**2) + np.mean(d2**2)) / 2
                cd_matrix[(i, j)] = cd
                if cd < cd_threshold:
                    union(parent, rank, i, j)
        
        # Group by root
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(parent, i)].append(i)
        
        # Keep one per cluster and track cluster sizes + full cluster info
        cluster_sizes = []
        cluster_info_list = []  # Full cluster info for clusters with >1 item
        
        for root, indices in clusters.items():
            cluster_sizes.append(len(indices))
            keep_idx = random.choice(indices)
            keep_uid = items_data[keep_idx][0]
            keep_list.append(keep_uid)
            
            # Build cluster info for clusters with multiple items
            if len(indices) > 1:
                cluster_members = []
                for idx in indices:
                    member_uid = items_data[idx][0]
                    if idx == keep_idx:
                        cluster_members.append({'uid': member_uid, 'status': 'kept'})
                    else:
                        key = (min(idx, keep_idx), max(idx, keep_idx))
                        cd_val = cd_matrix.get(key)
                        cluster_members.append({
                            'uid': member_uid,
                            'status': 'removed',
                            'cd_to_kept': cd_val
                        })
                        remove_list.append({'uid': member_uid, 'similar_to': keep_uid, 'cd': cd_val})
                
                cluster_info_list.append({
                    'kept': keep_uid,
                    'size': len(indices),
                    'members': cluster_members
                })
        
        # Add error_keep items as individual clusters of size 1
        cluster_sizes.extend([1] * len(error_keep))
        return keep_list + error_keep, remove_list, cluster_sizes, cluster_info_list
    
    # Process groups in parallel
    group_list = list(groups.values())
    print(f"Processing {len(group_list)} groups in parallel with {n_jobs} workers...")
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_group)(g, cd_threshold, n_points) for g in group_list
    )
    
    # Collect results and track group sizes
    keep_uids = set()
    remove_info = []
    group_sizes = []  # Track how many items kept per fingerprint group
    all_cd_cluster_sizes = []  # Track CD cluster sizes (before dedup within each cluster)
    all_cluster_info = []  # Full cluster info for clusters with >1 item
    
    for keep_list, remove_list, cd_cluster_sizes, cluster_info_list in results:
        keep_uids.update(keep_list)
        remove_info.extend(remove_list)
        if keep_list:  # Only count non-empty groups
            group_sizes.append(len(keep_list))
        all_cd_cluster_sizes.extend(cd_cluster_sizes)
        all_cluster_info.extend(cluster_info_list)
    
    # Group size distribution
    size_dist = Counter(group_sizes)
    cd_cluster_dist = Counter(all_cd_cluster_sizes)
    
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Total meshes processed: {len(mesh_files)}")
    print(f"Keep: {len(keep_uids)}")
    print(f"Remove (duplicates): {len(remove_info)}")
    print(f"Reduction: {len(remove_info) / len(mesh_files) * 100:.1f}%")
    print("-" * 60)
    print("STEP 1 - Fingerprint grouping (by area/volume):")
    print(f"  Total groups formed: {len(groups)}")
    orig_fp_sizes = Counter(len(g) for g in groups.values())
    for size in sorted(orig_fp_sizes.keys()):
        count = orig_fp_sizes[size]
        print(f"    {count} groups have {size} item(s)")
    print("-" * 60)
    print("STEP 2 - Chamfer Distance clustering (similar items grouped):")
    print(f"  Total clusters: {len(all_cd_cluster_sizes)}")
    for size in sorted(cd_cluster_dist.keys()):
        count = cd_cluster_dist[size]
        if size == 1:
            print(f"    {count} clusters have 1 item (unique, no duplicates)")
        else:
            removed = size - 1
            print(f"    {count} clusters have {size} similar items (keep 1, remove {removed} each)")
    print("=" * 60)
    
    # Sort clusters by size (largest first)
    all_cluster_info.sort(key=lambda x: x['size'], reverse=True)
    
    return {
        'keep': list(keep_uids),
        'duplicate_clusters': all_cluster_info  # Clusters with >1 item showing all members
    }


def save_filter_results(results: dict, output_path: str):
    """Save filter results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='filter', choices=['generate', 'filter'],
                        help="Mode: 'generate' meshes or 'filter' duplicates")
    parser.add_argument('--mesh-dir', type=str, default=MESH_OUTPUT_DIR,
                        help="Mesh directory for filtering")
    parser.add_argument('--output', type=str, default='filter_results.json',
                        help="Output file for filter results")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_all_meshes()
    elif args.mode == 'filter':
        results = filter_similar_objects(args.mesh_dir, max_samples=args.max_samples)
        save_filter_results(results, args.output)

