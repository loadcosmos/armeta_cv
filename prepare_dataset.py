"""
Dataset Preparation Script - Root Wrapper

This module provides access to dataset preparation functionality.
"""

# Import everything from the actual implementation
from kaggle.dataset.prepare_dataset import *

# If the module is run directly, execute the main function
if __name__ == '__main__':
    from kaggle.dataset.prepare_dataset import main
    main()
