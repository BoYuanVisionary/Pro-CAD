"""
Visualization utilities for CAD Agent.
Matplotlib-based rendering for headless servers.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageOps
from io import BytesIO
import trimesh
from typing import List, Tuple, Optional


def render_mesh_view(
    mesh: trimesh.Trimesh,
    elev: float,
    azim: float,
    img_size: int = 256,
    color: str = '#FFFF88'
) -> Image.Image:
    """
    Render mesh from a specific view using matplotlib.
    
    Args:
        mesh: Normalized trimesh object
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        img_size: Output image size in pixels
        color: Face color (hex string)
        
    Returns:
        PIL Image
    """
    fig = plt.figure(figsize=(4, 4), dpi=img_size // 4)
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Create polygon collection
    poly3d = [[vertices[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(poly3d, alpha=0.9, edgecolor='#333333', linewidth=0.1)
    collection.set_facecolor(color)
    ax.add_collection3d(collection)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Set view
    ax.view_init(elev=elev, azim=azim)
    
    # Clean up axes
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout(pad=0)
    
    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=img_size // 4, bbox_inches='tight', 
                pad_inches=0.05, facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    
    # Resize to exact size
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return img


def render_pointcloud_view(
    points: np.ndarray,
    elev: float,
    azim: float,
    img_size: int = 256,
    color: str = 'green',
    max_points: int = 2000
) -> Image.Image:
    """
    Render point cloud from a specific view using matplotlib.
    
    Args:
        points: Point cloud array of shape (N, 3)
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        img_size: Output image size in pixels
        color: Point color
        max_points: Maximum points to display (for performance)
        
    Returns:
        PIL Image
    """
    fig = plt.figure(figsize=(4, 4), dpi=img_size // 4)
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample points for faster rendering
    n_display = min(max_points, len(points))
    idx = np.random.choice(len(points), n_display, replace=False)
    pts = points[idx]
    
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=1, alpha=0.8)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Set view
    ax.view_init(elev=elev, azim=azim)
    
    # Clean up axes
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout(pad=0)
    
    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=img_size // 4, bbox_inches='tight',
                pad_inches=0.05, facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    
    # Resize to exact size
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return img


def get_view_angles(num_views: int) -> List[Tuple[float, float]]:
    """
    Get standard view angles for multi-view rendering.
    
    Args:
        num_views: Number of views (1, 4, or 6)
        
    Returns:
        List of (elevation, azimuth) tuples
    """
    if num_views == 1:
        return [(30, 45)]
    elif num_views == 4:
        return [(30, 45), (30, 135), (30, 225), (30, 315)]
    elif num_views == 6:
        return [(90, 0), (-90, 0), (0, 0), (0, 90), (0, 180), (0, 270)]
    else:
        return [(30, 45)]


def render_multiview(
    mesh: trimesh.Trimesh,
    num_views: int = 4,
    img_size: int = 256,
    color: str = '#FFFF88',
    border: int = 2
) -> Image.Image:
    """
    Render mesh from multiple viewpoints.
    
    Args:
        mesh: Normalized trimesh object
        num_views: Number of views (1, 4, or 6)
        img_size: Size of each view
        color: Face color
        border: Border width in pixels
        
    Returns:
        PIL Image grid
    """
    views = get_view_angles(num_views)
    
    # Render each view
    images = []
    for elev, azim in views:
        img = render_mesh_view(mesh, elev, azim, img_size, color=color)
        images.append(ImageOps.expand(img, border=border, fill='black'))
    
    # Create grid
    if num_views == 1:
        grid = images[0]
    elif num_views == 4:
        grid = Image.fromarray(np.vstack((
            np.hstack((np.array(images[0]), np.array(images[1]))),
            np.hstack((np.array(images[2]), np.array(images[3])))
        )))
    elif num_views == 6:
        grid = Image.fromarray(np.vstack((
            np.hstack((np.array(images[0]), np.array(images[1]), np.array(images[2]))),
            np.hstack((np.array(images[3]), np.array(images[4]), np.array(images[5])))
        )))
    else:
        grid = images[0]
    
    return grid


def render_pointcloud_multiview(
    points: np.ndarray,
    num_views: int = 4,
    img_size: int = 256,
    color: str = 'green',
    border: int = 2
) -> Image.Image:
    """
    Render point cloud from multiple viewpoints.
    
    Args:
        points: Point cloud array of shape (N, 3)
        num_views: Number of views (1 or 4)
        img_size: Size of each view
        color: Point color
        border: Border width in pixels
        
    Returns:
        PIL Image grid
    """
    views = get_view_angles(num_views)
    
    images = []
    for elev, azim in views:
        img = render_pointcloud_view(points, elev, azim, img_size, color=color)
        images.append(ImageOps.expand(img, border=border, fill='black'))
    
    # Create grid
    if num_views == 1:
        grid = images[0]
    elif num_views >= 4:
        grid = Image.fromarray(np.vstack((
            np.hstack((np.array(images[0]), np.array(images[1]))),
            np.hstack((np.array(images[2]), np.array(images[3])))
        )))
    else:
        grid = images[0]
    
    return grid


def render_comparison(
    gt_mesh: Optional[trimesh.Trimesh] = None,
    gen_mesh: Optional[trimesh.Trimesh] = None,
    gt_points: Optional[np.ndarray] = None,
    gen_points: Optional[np.ndarray] = None,
    img_size: int = 300,
    view: Tuple[float, float] = (30, 45)
) -> Image.Image:
    """
    Render side-by-side comparison of ground truth and generated shapes.
    
    Args:
        gt_mesh: Ground truth mesh
        gen_mesh: Generated mesh
        gt_points: Ground truth point cloud
        gen_points: Generated point cloud
        img_size: Size of each view
        view: (elevation, azimuth) for rendering
        
    Returns:
        PIL Image with side-by-side comparison
    """
    elev, azim = view
    images = []
    
    if gt_mesh is not None:
        gt_img = render_mesh_view(gt_mesh, elev, azim, img_size, color='#88FF88')
        images.append(np.array(gt_img))
    elif gt_points is not None:
        gt_img = render_pointcloud_view(gt_points, elev, azim, img_size, color='green')
        images.append(np.array(gt_img))
    
    if gen_mesh is not None:
        gen_img = render_mesh_view(gen_mesh, elev, azim, img_size, color='#88CCFF')
        images.append(np.array(gen_img))
    elif gen_points is not None:
        gen_img = render_pointcloud_view(gen_points, elev, azim, img_size, color='blue')
        images.append(np.array(gen_img))
    
    if len(images) == 2:
        comparison = Image.fromarray(np.hstack(images))
    elif len(images) == 1:
        comparison = Image.fromarray(images[0])
    else:
        comparison = Image.new('RGB', (img_size * 2, img_size), 'white')
    
    return comparison

