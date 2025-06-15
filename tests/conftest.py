"""
Shared pytest fixtures and configuration for all tests.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Provide a mock configuration dictionary for testing.
    
    Returns:
        Dict[str, Any]: Mock configuration settings
    """
    return {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 10,
        'embedding_size': 64,
        'regularization': 0.001,
        'validation_split': 0.2,
        'random_seed': 42
    }


@pytest.fixture
def sample_data_dir(temp_dir: Path) -> Path:
    """
    Create a sample data directory with test files.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the sample data directory
    """
    data_dir = temp_dir / "sample_data"
    data_dir.mkdir(exist_ok=True)
    
    # Create some sample files
    (data_dir / "users.csv").write_text("user_id,age,gender\n1,25,M\n2,30,F\n3,35,M")
    (data_dir / "items.csv").write_text("item_id,category,price\n1,electronics,100\n2,books,20\n3,electronics,200")
    (data_dir / "ratings.csv").write_text("user_id,item_id,rating,timestamp\n1,1,5,1000\n1,2,3,1001\n2,3,4,1002")
    
    return data_dir


@pytest.fixture
def sample_sparse_matrix():
    """
    Create a sample sparse matrix for testing recommender systems.
    
    Returns:
        scipy.sparse.csr_matrix: Sample user-item interaction matrix
    """
    from scipy.sparse import csr_matrix
    
    # Create a small user-item matrix (5 users x 6 items)
    data = np.array([5, 3, 4, 5, 2, 1, 4, 3, 5])
    row_indices = np.array([0, 0, 1, 2, 2, 3, 3, 4, 4])
    col_indices = np.array([0, 2, 3, 0, 4, 1, 5, 2, 3])
    
    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(5, 6))
    return matrix


@pytest.fixture
def sample_dataframe():
    """
    Create a sample pandas DataFrame for testing.
    
    Returns:
        pd.DataFrame: Sample ratings dataframe
    """
    data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5],
        'item_id': [1, 3, 4, 1, 5, 2, 6, 3, 4],
        'rating': [5.0, 3.0, 4.0, 5.0, 2.0, 1.0, 4.0, 3.0, 5.0],
        'timestamp': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_recommender():
    """
    Create a mock recommender object for testing.
    
    Returns:
        Mock: Mock recommender with basic methods
    """
    recommender = Mock()
    recommender.fit = MagicMock(return_value=None)
    recommender.predict = MagicMock(return_value=np.array([0.8, 0.6, 0.9, 0.4, 0.7]))
    recommender.recommend = MagicMock(return_value=(np.array([2, 0, 4]), np.array([0.9, 0.8, 0.7])))
    recommender.get_item_weights = MagicMock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    recommender.save_model = MagicMock(return_value=None)
    recommender.load_model = MagicMock(return_value=None)
    return recommender


@pytest.fixture
def mock_evaluator():
    """
    Create a mock evaluator object for testing.
    
    Returns:
        Mock: Mock evaluator with evaluation methods
    """
    evaluator = Mock()
    evaluator.evaluate_recommender = MagicMock(return_value={
        'precision': 0.75,
        'recall': 0.68,
        'f1_score': 0.71,
        'ndcg': 0.82,
        'map': 0.79
    })
    return evaluator


@pytest.fixture
def mock_data_reader():
    """
    Create a mock data reader for testing.
    
    Returns:
        Mock: Mock data reader with load methods
    """
    reader = Mock()
    reader.load_data = MagicMock(return_value={
        'URM_train': Mock(shape=(100, 200)),
        'URM_test': Mock(shape=(100, 200)),
        'URM_validation': Mock(shape=(100, 200)),
        'ICM': Mock(shape=(200, 50)),
        'UCM': Mock(shape=(100, 30))
    })
    return reader


@pytest.fixture
def mock_similarity_matrix():
    """
    Create a mock similarity matrix for testing.
    
    Returns:
        np.ndarray: Sample similarity matrix
    """
    n_items = 6
    similarity = np.random.rand(n_items, n_items)
    # Make it symmetric
    similarity = (similarity + similarity.T) / 2
    # Set diagonal to 1
    np.fill_diagonal(similarity, 1.0)
    return similarity


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Automatically reset random seeds before each test for reproducibility.
    """
    np.random.seed(42)
    import random
    random.seed(42)
    
    # If tensorflow is available, set its seed too
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except ImportError:
        pass


@pytest.fixture
def capture_logs(caplog):
    """
    Fixture to capture log messages during tests.
    
    Args:
        caplog: pytest's built-in log capture fixture
        
    Returns:
        caplog: Configured log capture
    """
    caplog.set_level("DEBUG")
    return caplog


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path):
    """
    Create a mock model checkpoint file.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the mock checkpoint file
    """
    checkpoint_path = temp_dir / "model_checkpoint.pkl"
    
    # Create a simple mock checkpoint
    import pickle
    checkpoint_data = {
        'model_state': {'layer1': np.random.rand(10, 10), 'layer2': np.random.rand(10, 5)},
        'optimizer_state': {'learning_rate': 0.01, 'iteration': 1000},
        'metrics': {'train_loss': 0.05, 'val_loss': 0.08}
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    return checkpoint_path


@pytest.fixture
def environment_variables():
    """
    Temporarily set environment variables for testing.
    
    Yields:
        Dict[str, str]: Dictionary to set environment variables
    """
    original_env = os.environ.copy()
    test_env = {}
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    
    # Apply test environment variables
    os.environ.update(test_env)