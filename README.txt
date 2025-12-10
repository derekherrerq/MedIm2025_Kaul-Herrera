PREREQUISITES
-------------
- Python 3.12 (recommended and tested)
- pip package installer
- Approximately 50MB disk space
- Verify installation: python --version


INSTALLATION
------------
1. Create a virtual environment (recommended):

   py -3.12 -m venv .venv

2. Activate the virtual environment:

   Windows:
   .\.venv\Scripts\activate

   Your terminal prompt should now begin with: (.venv)

3. Install required packages:

   pip install --upgrade pip
   pip install -r requirements.txt

   Installs: numpy, matplotlib, scikit-image, scipy, streamlit, pandas


DATA / DATASETS
---------------
N/A - This simulation generates synthetic x-ray images and does not require external datasets.


RUNNING THE CODE
================

INTERACTIVE GUI (PRIMARY APPLICATION)
--------------------------------------
Run the Streamlit-based GUI from the project root:

streamlit run gui_app.py

FEATURES:
- Opens browser automatically at http://localhost:8501
- Adjust parameters in left sidebar (energy, distances, angles)
- Click "Run Simulation" to generate x-rays
- View results in tabs: Images, Profiles, Metrics
- Experiment with 2D test phantom or 3D leg phantom with fractures
- Click "Run Test Simulator" to verify installation
- Click "Run Batch Analysis" from sidebar to sweep parameters


RUNNING INDIVIDUAL SCRIPTS
--------------------------
The following scripts can be run directly from the command line:

- python test_simulator.py
- python batch_analysis.py
- python xray_simulator.py
- python visualization.py




MODEL EVALUATION
----------------
Image quality evaluated via built-in metrics:
- Contrast
- Sharpness (Variance of Laplacian)
- Edge Strength

Run batch_analysis.py to compare metrics across different simulation parameters.


PROJECT STRUCTURE
-----------------
MedIm2025_Kaul-Herrera/
    gui_app.py              Main Streamlit interface
    batch_analysis.py       Batch simulation driver
    test_simulator.py       Simulator test script
    xray_simulator.py       Core x-ray simulation logic
    visualization.py        Plotting utilities
    image_metrics.py        Metric calculations

    ProjectFunctions/
        image_metrics.py
        visualization.py
        __pycache__/

    requirements.txt
    README.md
    .venv/                  Virtual environment (ignored by Git)


GIT NOTES
---------
- Do not commit .venv/ or __pycache__/ directories
- A .gitignore file is included to prevent committing temporary or environment files
