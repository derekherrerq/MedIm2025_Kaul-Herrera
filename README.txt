PREREQUISITES
-------------
- Python 3.12 (recommended and tested)
- pip package installer
- Approximately 50MB disk space
- Verify installation: python --version

INSTALLATION
------------
1. Create a virtual environment (recommended):

   Windows:
       py -3.12 -m venv .venv
   
   macOS / Linux:
       python3 -m venv .venv

2. Activate the virtual environment:

   Windows:
       .\.venv\Scripts\activate
   
   macOS / Linux:
       source .venv/bin/activate
   
   Your terminal prompt should now begin with (.venv)

3. Install required packages:

   pip install --upgrade pip
   pip install -r requirements.txt
   
   This installs: numpy, matplotlib, scikit-image, scipy, streamlit, pandas

DATA / DATASETS
---------------
N/A - This simulation generates synthetic x-ray images and does not require 
external datasets.

RUNNING THE CODE
================

INTERACTIVE GUI (PRIMARY APPLICATION)
--------------------------------------
Run the Streamlit-based GUI from the project root:

    streamlit run MedIm2025_Code_Kaul-Herrera/ProjectFunctions/gui_app.py

FEATURES:
- Opens browser automatically at http://localhost:8501
- Adjust parameters in left sidebar (energy, distances, angles)
- Click "Run Simulation" to generate x-rays
- View results in tabs: Images, Profiles, Metrics
- Experiment with 2D test phantom or 3D leg phantom with fractures
- Click "Run Test Simulator" to verify installation
- Click "Run Batch Analysis" from sidebar to sweep parameters

RUNNING INDIVIDUAL SCRIPTS
---------------------------
The following scripts can be run directly from the command line:

    python MedIm2025_Code_Kaul-Herrera/ProjectFunctions/test_simulator.py
    python MedIm2025_Code_Kaul-Herrera/ProjectFunctions/batch_analysis.py
    python MedIm2025_Code_Kaul-Herrera/ProjectFunctions/xray_simulator.py
    python MedIm2025_Code_Kaul-Herrera/ProjectFunctions/visualization.py

MODEL EVALUATION
----------------
Image quality evaluated via built-in metrics:
- Contrast
- Sharpness (Variance of Laplacian)
- Edge Strength

Run batch_analysis.py to compare metrics across different simulation parameters.

GIT NOTES
---------
- Do not commit .venv/ or __pycache__/ directories
- A .gitignore file is included to prevent committing temporary or 
  environment files