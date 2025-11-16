"""
Mobile Document Scanner App - Root Wrapper

This module provides access to the Streamlit mobile application.
"""

# Import everything from the actual implementation
from local.app.mobile_app import *

# If the module is run directly, this won't work with streamlit
# Users should run: streamlit run mobile_app.py
if __name__ == '__main__':
    print("Please run this app with: streamlit run mobile_app.py")
