"""
Mesh utilities for CAD Agent.
Functions for loading, normalizing, and sampling meshes.
"""

import numpy as np
import trimesh
import cadquery as cq
from typing import Tuple, Optional


def load_mesh_from_stl(stl_path: str) -> trimesh.Trimesh:
    """
    Load mesh from STL file.
    
    Args:
        stl_path: Path to the STL file
        
    Returns:
        trimesh.Trimesh object
    """
    mesh = trimesh.load(stl_path)
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalize mesh to unit cube centered at (0.5, 0.5, 0.5).
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Normalized trimesh object (copy of input)
    """
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_translation([0.5, 0.5, 0.5])
    return mesh


def mesh_to_pointcloud(mesh: trimesh.Trimesh, n_points: int = 8192) -> np.ndarray:
    """
    Sample point cloud from mesh surface.
    
    Args:
        mesh: trimesh object
        n_points: Number of points to sample
        
    Returns:
        Point cloud as numpy array of shape (n_points, 3)
    """
    points = mesh.sample(n_points)
    return points


def cadquery_to_mesh(code: str) -> trimesh.Trimesh:
    """
    Execute CadQuery code and convert result to trimesh.
    
    The code must define a variable 'r' containing the CadQuery result.
    
    Args:
        code: CadQuery Python code string
        
    Returns:
        trimesh.Trimesh object
        
    Raises:
        ValueError: If code doesn't define 'r' variable
    """
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


def cadquery_result_to_mesh(cq_result) -> trimesh.Trimesh:
    """
    Convert CadQuery result object to trimesh.
    
    Args:
        cq_result: CadQuery Workplane result
        
    Returns:
        trimesh.Trimesh object
    """
    compound = cq_result.val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    
    mesh = trimesh.Trimesh(
        vertices=[(v.x, v.y, v.z) for v in vertices],
        faces=faces
    )
    return mesh


def load_and_normalize_mesh(stl_path: str) -> trimesh.Trimesh:
    """
    Load mesh from STL and normalize to unit cube.
    
    Args:
        stl_path: Path to STL file
        
    Returns:
        Normalized trimesh object
    """
    mesh = load_mesh_from_stl(stl_path)
    return normalize_mesh(mesh)


def mesh_to_normalized_pointcloud(
    mesh: trimesh.Trimesh,
    n_points: int = 8192,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert mesh to normalized point cloud.
    
    Args:
        mesh: trimesh object
        n_points: Number of points to sample
        normalize: Whether to normalize mesh first
        
    Returns:
        Point cloud as numpy array
    """
    if normalize:
        mesh = normalize_mesh(mesh)
    return mesh_to_pointcloud(mesh, n_points)


def stl_to_pointcloud(
    stl_path: str,
    n_points: int = 8192
) -> np.ndarray:
    """
    Load STL file and convert directly to normalized point cloud.
    
    Args:
        stl_path: Path to STL file
        n_points: Number of points to sample
        
    Returns:
        Point cloud as numpy array
    """
    mesh = load_and_normalize_mesh(stl_path)
    return mesh_to_pointcloud(mesh, n_points)


def code_to_pointcloud(
    code: str,
    n_points: int = 8192
) -> np.ndarray:
    """
    Execute CadQuery code and convert to normalized point cloud.
    
    Args:
        code: CadQuery Python code string
        n_points: Number of points to sample
        
    Returns:
        Point cloud as numpy array
    """
    mesh = cadquery_to_mesh(code)
    mesh = normalize_mesh(mesh)
    return mesh_to_pointcloud(mesh, n_points)

