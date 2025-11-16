"""
YOLOv8s Training Script - Root Wrapper

This module provides access to model training functionality.
"""

# Import everything from the actual implementation
from kaggle.training.train_yolov8s_optimized import *

# If the module is run directly, execute the main function
if __name__ == '__main__':
    from kaggle.training.train_yolov8s_optimized import main
    main()
