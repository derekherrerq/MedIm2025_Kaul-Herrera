"""
quick test script
verifies that the x-ray simulator is working correctly
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

print("Testing Virtual X-Ray Simulator...")
print("="*60)

# test 1: import modules
print("\n1. Testing imports...")
try:
    from ProjectFunctions.xray_simulator import XRaySimulator
    from ProjectFunctions.image_metrics import ImageMetrics
    from ProjectFunctions.visualization import XRayVisualizer
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# test 2: create simulator
print("\n2. Creating simulator...")
try:
    simulator = XRaySimulator(image_size=(256, 256))
    print("   ✓ Simulator created successfully")
except Exception as e:
    print(f"   ✗ Simulator creation failed: {e}")
    exit(1)

# test 3: test 2d phantom
print("\n3. Testing 2D phantom simulation...")
try:
    phantom, xray_image = simulator.simulate_xray(
        phantom_type='2d_test',
        energy=60.0,
        noise_level=0.01
    )
    print(f"   ✓ 2D simulation successful")
    print(f"     Phantom shape: {phantom.shape}")
    print(f"     X-ray image shape: {xray_image.shape}")
    print(f"     X-ray intensity range: [{xray_image.min():.3f}, {xray_image.max():.3f}]")
except Exception as e:
    print(f"   ✗ 2D simulation failed: {e}")
    exit(1)

# test 4: test 3d leg phantom
print("\n4. Testing 3D leg phantom simulation...")
try:
    phantom_3d, xray_3d = simulator.simulate_xray(
        phantom_type='3d_leg',
        energy=60.0,
        fracture=True,
        fracture_width=2.0,
        noise_level=0.01
    )
    print(f"   ✓ 3D simulation successful")
    print(f"     X-ray image shape: {xray_3d.shape}")
    print(f"     X-ray intensity range: [{xray_3d.min():.3f}, {xray_3d.max():.3f}]")
except Exception as e:
    print(f"   ✗ 3D simulation failed: {e}")
    exit(1)

# test 5: calculate metrics
print("\n5. Testing image metrics...")
try:
    metrics = ImageMetrics.get_all_metrics(xray_image)
    print(f"   ✓ Metrics calculated successfully")
    print(f"     Contrast: {metrics['contrast']:.4f}")
    print(f"     Sharpness: {metrics['variance_of_laplacian']:.4f}")
    print(f"     Edge strength: {metrics['edge_strength']:.4f}")
except Exception as e:
    print(f"   ✗ Metrics calculation failed: {e}")
    exit(1)

# test 6: test visualization
print("\n6. Testing visualization...")
try:
    visualizer = XRayVisualizer()
    fig = visualizer.plot_phantom_and_xray(phantom, xray_image, "Test X-Ray")
    plt.savefig('test_output.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Visualization successful")
    print(f"     Test image saved as 'test_output.png'")
except Exception as e:
    print(f"   ✗ Visualization failed: {e}")
    exit(1)

# test 7: test parameter variations
print("\n7. Testing parameter variations...")
try:
    energies = [30, 60, 100]
    for energy in energies:
        _, img = simulator.simulate_xray(
            phantom_type='2d_test',
            energy=energy,
            noise_level=0.01
        )
        contrast = ImageMetrics.calculate_contrast(img)
        print(f"     Energy {energy} keV → Contrast: {contrast:.4f}")
    print(f"   ✓ Parameter variations working correctly")
except Exception as e:
    print(f"   ✗ Parameter variation test failed: {e}")
    exit(1)

# test 8: test attenuation coefficients
print("\n8. Testing attenuation coefficient lookup...")
try:
    materials = ['air', 'soft_tissue', 'bone']
    test_energy = 60.0
    for material in materials:
        mu = simulator.get_attenuation_coefficient(material, test_energy)
        print(f"     {material:12s} @ {test_energy}keV: μ = {mu:.5f} cm⁻¹")
    print(f"   ✓ Attenuation coefficients retrieved successfully")
except Exception as e:
    print(f"   ✗ Attenuation coefficient test failed: {e}")
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nThe simulator is working correctly.")
print("You can now run:")
print("  - 'streamlit run gui_app.py' for interactive GUI")
print("  - 'python batch_analysis.py' for comprehensive analysis")
print("="*60)