"""
Validation tests to ensure the testing infrastructure is properly set up.
"""
import pytest
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


class TestInfrastructureValidation:
    """Test suite to validate the testing infrastructure setup."""
    
    def test_pytest_is_importable(self):
        """Test that pytest is properly installed and importable."""
        import pytest
        assert pytest.__version__
        
    def test_pytest_cov_is_importable(self):
        """Test that pytest-cov is properly installed and importable."""
        import pytest_cov
        assert pytest_cov.__version__
        
    def test_pytest_mock_is_importable(self):
        """Test that pytest-mock is properly installed and importable."""
        import pytest_mock
        # pytest_mock doesn't expose __version__, just check it's importable
        assert pytest_mock is not None
    
    def test_project_root_in_path(self):
        """Test that the project root is in Python path."""
        project_root = Path(__file__).parent.parent
        assert str(project_root) in sys.path or str(project_root.absolute()) in sys.path
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that the temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test file creation in temp dir
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_mock_config_fixture(self, mock_config):
        """Test that the mock_config fixture provides expected structure."""
        assert isinstance(mock_config, dict)
        assert 'learning_rate' in mock_config
        assert 'batch_size' in mock_config
        assert 'epochs' in mock_config
        assert mock_config['random_seed'] == 42
    
    def test_sample_data_dir_fixture(self, sample_data_dir):
        """Test that the sample_data_dir fixture creates expected files."""
        assert sample_data_dir.exists()
        assert (sample_data_dir / "users.csv").exists()
        assert (sample_data_dir / "items.csv").exists()
        assert (sample_data_dir / "ratings.csv").exists()
        
        # Check content
        users_df = pd.read_csv(sample_data_dir / "users.csv")
        assert len(users_df) == 3
        assert list(users_df.columns) == ['user_id', 'age', 'gender']
    
    def test_sample_sparse_matrix_fixture(self, sample_sparse_matrix):
        """Test that the sample_sparse_matrix fixture works correctly."""
        from scipy.sparse import issparse
        
        assert issparse(sample_sparse_matrix)
        assert sample_sparse_matrix.shape == (5, 6)
        assert sample_sparse_matrix.nnz == 9  # number of non-zero elements
    
    def test_sample_dataframe_fixture(self, sample_dataframe):
        """Test that the sample_dataframe fixture provides expected data."""
        assert isinstance(sample_dataframe, pd.DataFrame)
        assert len(sample_dataframe) == 9
        assert list(sample_dataframe.columns) == ['user_id', 'item_id', 'rating', 'timestamp']
        assert sample_dataframe['rating'].min() >= 1.0
        assert sample_dataframe['rating'].max() <= 5.0
    
    def test_mock_recommender_fixture(self, mock_recommender):
        """Test that the mock_recommender fixture has expected methods."""
        assert hasattr(mock_recommender, 'fit')
        assert hasattr(mock_recommender, 'predict')
        assert hasattr(mock_recommender, 'recommend')
        
        # Test method calls
        mock_recommender.fit(None)
        predictions = mock_recommender.predict(None)
        assert isinstance(predictions, np.ndarray)
        
        items, scores = mock_recommender.recommend(None)
        assert isinstance(items, np.ndarray)
        assert isinstance(scores, np.ndarray)
    
    def test_capture_logs_fixture(self, capture_logs):
        """Test that log capturing works correctly."""
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("Test info message")
        logger.debug("Test debug message")
        
        assert "Test info message" in capture_logs.text
        assert "Test debug message" in capture_logs.text
    
    @pytest.mark.parametrize("marker", ["unit", "integration", "slow"])
    def test_custom_markers_defined(self, marker):
        """Test that custom pytest markers are properly defined."""
        # Custom markers should be usable without error
        try:
            getattr(pytest.mark, marker)
        except AttributeError:
            pytest.fail(f"Marker '{marker}' is not defined")
    
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        # This test verifies that the test can run with coverage
        # The actual coverage configuration is tested by running pytest with coverage
        import coverage
        assert coverage.__version__
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True


class TestPytestCommands:
    """Test that pytest commands are properly configured."""
    
    def test_pytest_runs_from_project_root(self):
        """Test that pytest can be run from project root."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").exists()
    
    def test_test_discovery_patterns(self):
        """Test that test files follow naming conventions."""
        test_file = Path(__file__)
        assert test_file.name.startswith("test_") or test_file.name.endswith("_test.py")


if __name__ == "__main__":
    # Allow running this file directly for quick validation
    pytest.main([__file__, "-v"])