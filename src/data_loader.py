"""
Data loading utilities for CAD Agent.
Functions for loading test/train data and filtering samples.
"""

import os
import pickle
from typing import List, Dict, Optional


def load_pickle_data(data_path: str) -> List[Dict]:
    """
    Load data from a pickle file.
    
    Args:
        data_path: Path to the pickle file (e.g., test.pkl, train.pkl)
        
    Returns:
        List of sample dictionaries
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def filter_samples_with_mesh(
    samples: List[Dict],
    mesh_dir: str,
    mesh_ext: str = ".stl"
) -> List[Dict]:
    """
    Filter samples that have corresponding mesh files.
    
    Args:
        samples: List of sample dictionaries with 'uid' key
        mesh_dir: Directory containing mesh files
        mesh_ext: Mesh file extension (default: .stl)
        
    Returns:
        Filtered list of samples with existing mesh files
    """
    valid_samples = []
    for sample in samples:
        uid = sample['uid']
        mesh_path = os.path.join(mesh_dir, f"{uid}{mesh_ext}")
        if os.path.exists(mesh_path):
            valid_samples.append(sample)
    return valid_samples


def filter_samples_with_code(
    samples: List[Dict],
    code_dir: str,
    code_ext: str = ".py"
) -> List[Dict]:
    """
    Filter samples that have corresponding code files.
    
    Args:
        samples: List of sample dictionaries with 'uid' key
        code_dir: Directory containing code files
        code_ext: Code file extension (default: .py)
        
    Returns:
        Filtered list of samples with existing code files
    """
    valid_samples = []
    for sample in samples:
        uid = sample['uid']
        code_path = os.path.join(code_dir, f"{uid}{code_ext}")
        if os.path.exists(code_path):
            valid_samples.append(sample)
    return valid_samples


def load_code_file(code_dir: str, uid: str, ext: str = ".py") -> Optional[str]:
    """
    Load code from a file.
    
    Args:
        code_dir: Directory containing code files
        uid: Sample UID
        ext: File extension (default: .py)
        
    Returns:
        Code string or None if file not found
    """
    code_path = os.path.join(code_dir, f"{uid}{ext}")
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            return f.read()
    return None


def get_sample_by_uid(samples: List[Dict], uid: str) -> Optional[Dict]:
    """
    Get a sample by its UID.
    
    Args:
        samples: List of sample dictionaries
        uid: Sample UID to find
        
    Returns:
        Sample dictionary or None if not found
    """
    for sample in samples:
        if sample['uid'] == uid:
            return sample
    return None


class Text2CADDataset:
    """
    Dataset class for Text2CAD data.
    Provides easy access to samples, descriptions, meshes, and code.
    """
    
    def __init__(
        self,
        data_path: str,
        mesh_dir: Optional[str] = None,
        code_dir: Optional[str] = None,
        filter_valid: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to pickle file (test.pkl or train.pkl)
            mesh_dir: Directory containing ground truth mesh files
            code_dir: Directory containing CadQuery code files
            filter_valid: Whether to filter samples with existing files
        """
        self.data_path = data_path
        self.mesh_dir = mesh_dir
        self.code_dir = code_dir
        
        # Load data
        self.samples = load_pickle_data(data_path)
        
        # Filter if requested
        if filter_valid:
            if mesh_dir:
                self.samples = filter_samples_with_mesh(self.samples, mesh_dir)
            if code_dir:
                self.samples = filter_samples_with_code(self.samples, code_dir)
        
        # Build UID index
        self._uid_index = {s['uid']: i for i, s in enumerate(self.samples)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
    def get_by_uid(self, uid: str) -> Optional[Dict]:
        """Get sample by UID."""
        idx = self._uid_index.get(uid)
        if idx is not None:
            return self.samples[idx]
        return None
    
    def get_description(self, uid: str) -> Optional[str]:
        """Get description for a sample."""
        sample = self.get_by_uid(uid)
        return sample.get('description') if sample else None
    
    def get_mesh_path(self, uid: str) -> Optional[str]:
        """Get mesh file path for a sample."""
        if not self.mesh_dir:
            return None
        path = os.path.join(self.mesh_dir, f"{uid}.stl")
        return path if os.path.exists(path) else None
    
    def get_code(self, uid: str) -> Optional[str]:
        """Get CadQuery code for a sample."""
        if not self.code_dir:
            return None
        return load_code_file(self.code_dir, uid)
    
    def get_uids(self) -> List[str]:
        """Get all UIDs in the dataset."""
        return list(self._uid_index.keys())


# Default paths (relative to CAD_Agent directory)
DEFAULT_TEST_PATH = "./data/text2cad/test.pkl"
DEFAULT_TRAIN_PATH = "./data/text2cad/train.pkl"
DEFAULT_MESH_DIR = "./data/text2cad/deepcad_test_mesh"
DEFAULT_CODE_DIR = "./data/text2cad/cadquery"


def load_test_dataset(
    data_path: str = DEFAULT_TEST_PATH,
    mesh_dir: str = DEFAULT_MESH_DIR,
    code_dir: str = DEFAULT_CODE_DIR
) -> Text2CADDataset:
    """
    Convenience function to load the test dataset with default paths.
    
    Args:
        data_path: Path to test.pkl
        mesh_dir: Directory with ground truth meshes
        code_dir: Directory with CadQuery code
        
    Returns:
        Text2CADDataset instance
    """
    return Text2CADDataset(
        data_path=data_path,
        mesh_dir=mesh_dir,
        code_dir=code_dir,
        filter_valid=True
    )

