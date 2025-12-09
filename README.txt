PREREQUISITES
-------------
- Python 3.8+ (verify: python --version)
- pip package installer
- 50MB disk space

INSTALLATION
------------
pip install -r requirements.txt

Installs: numpy, matplotlib, scikit-image, scipy, streamlit, pandas

DATA/DATASETS
-------------
N/A - No external datasets required since it's simulation that generates synthetic x-ray images

RUNNING THE CODE
================

Interactive GUI (Everything in One Place)
------------------------------------------------------------------
streamlit run ProjectFunctions/gui_app.py

USE WHEN:
- Exploring how individual parameters affect x-ray images in real-time
- Creating specific x-ray images with custom settings
- Running test simulator or batch analysis

FEATURES:
- Opens browser automatically at http://localhost:8501
- Adjust parameters in left sidebar (energy, distances, angles)
- Click "Run Simulation" to generate x-rays
- View results in tabs: Images, Profiles, Metrics
- Experiment with 2D test phantom or 3D leg phantom with fractures
- Click "Run Test Simulator" button in sidebar to verify installation
- Click "Run Batch Analysis" button in sidebar for comprehensive parameter sweeps

MODEL TRAINING
--------------
N/A - This is a physics-based simulation. No training required. Results are deterministic based on Beer-Lambert law.

MODEL EVALUATION
----------------
N/A - Image quality evaluated via built-in metrics:
- Contrast, Sharpness (Variance of Laplacian), Edge Strength
- Run batch_analysis.py to see metric comparisons across parameters