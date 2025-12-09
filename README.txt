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

VERIFICATION
------------
Test installation by importing modules:

python -c "from xray_simulator import XRaySimulator; print('✓ Installation successful')"

Expected output: "✓ Installation successful"

RUNNING THE CODE
================

OPTION A: Interactive GUI (Exploration & Individual Testing)
-------------------------------------------------------------
streamlit run gui_app.py

USE WHEN:
- Exploring how individual parameters affect x-ray images in real-time
- Creating specific x-ray images with custom settings

FEATURES:
- Opens browser automatically at http://localhost:8501
- Adjust parameters in left sidebar (energy, distances, angles)
- Click "Run Simulation" to generate x-rays
- View results in tabs: Images, Profiles, Metrics
- Experiment with 2D test phantom or 3D leg phantom with fractures

OPTION B: Batch Analysis (Comprehensive Parameter Studies)
-----------------------------------------------------------
python batch_analysis.py

USE WHEN:
- Analyzing how parameters systematically affect results across multiple settings
- Generating comparison plots and CSV data for reports

OUTPUTS:
- Creates 'results/' folder with ~17 output files
- Generates parameter sweep analyses (energy, distance, angle, fracture)
- Creates CSV files with quantitative metrics
- Runtime: 2-5 minutes

Expected outputs in results/:
- sample_2d_test.png, sample_3d_leg_*.png (x-ray images)
- energy_sweep_analysis.png + metrics.csv
- distance_sweep_comparison.png + metrics (png + csv)
- angle_sweep_comparison.png + metrics (png + csv)
- fracture_width/angle_comparison.png + metrics.csv

OPTION C: Requirement Validation
-------------------------------------------------------
Validate specific project requirements:

python test_phantom_validation.py

USE WHEN:
- Demonstrating Requirement #4 (test phantom validation)
- Need explicit proof that algorithms work with distance/angle/attenuation changes

OUTPUTS:
- Creates 'validation_results/' folder

MODEL TRAINING
--------------
N/A - This is a physics-based simulation. No training required. Results are deterministic based on Beer-Lambert law.

MODEL EVALUATION
----------------
N/A - Image quality evaluated via built-in metrics:
- Contrast, Sharpness (Variance of Laplacian), Edge Strength
- Run batch_analysis.py to see metric comparisons across parameters