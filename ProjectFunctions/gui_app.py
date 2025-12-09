"""
streamlit gui for virtual x-ray simulation
interactive interface for exploring x-ray imaging parameters
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from xray_simulator import XRaySimulator
from image_metrics import ImageMetrics
from visualization import XRayVisualizer
import io


# page configuration
st.set_page_config(
    page_title="Virtual X-Ray Simulator",
    page_icon="X",
    layout="wide"
)

# initialize simulator
@st.cache_resource
def get_simulator():
    return XRaySimulator(image_size=(512, 512))

simulator = get_simulator()
visualizer = XRayVisualizer()

# title and description
st.title("Virtual X-Ray Simulation System")
st.markdown("""
This application simulates x-ray imaging using the Beer-Lambert law. 
Adjust parameters to see how they affect the resulting x-ray image and quality metrics.
""")

# sidebar for parameters
st.sidebar.header("Simulation Parameters")

# phantom selection
phantom_type = st.sidebar.selectbox(
    "Phantom Type",
    ["2D Test Phantom", "3D Leg Phantom"],
    help="Choose between a 2D test phantom with geometric shapes or a 3D cylindrical leg phantom"
)

# x-ray energy
energy = st.sidebar.slider(
    "X-Ray Energy (keV)",
    min_value=20.0,
    max_value=150.0,
    value=60.0,
    step=5.0,
    help="Higher energies penetrate more but reduce contrast"
)

# geometry parameters
st.sidebar.subheader("Geometric Parameters")

source_distance = st.sidebar.slider(
    "Source Distance (cm)",
    min_value=50.0,
    max_value=200.0,
    value=100.0,
    step=10.0,
    help="Distance from x-ray source to reference plane"
)

object_distance = st.sidebar.slider(
    "Object Distance (cm)",
    min_value=20.0,
    max_value=150.0,
    value=50.0,
    step=5.0,
    help="Distance from source to object (phantom)"
)

film_distance = st.sidebar.slider(
    "Film Distance (cm)",
    min_value=60.0,
    max_value=250.0,
    value=150.0,
    step=10.0,
    help="Distance from source to detector/film"
)

# beam angle
beam_angle = st.sidebar.slider(
    "Beam Angle (degrees)",
    min_value=-45.0,
    max_value=45.0,
    value=0.0,
    step=5.0,
    help="Angle of x-ray beam (0 = perpendicular)"
)

# noise level
noise_level = st.sidebar.slider(
    "Noise Level",
    min_value=0.0,
    max_value=0.05,
    value=0.01,
    step=0.005,
    help="Amount of noise to add (simulates quantum noise)"
)

# fracture parameters (only for 3d leg phantom)
fracture = False
fracture_angle = 0.0
fracture_width = 2.0

if phantom_type == "3D Leg Phantom":
    st.sidebar.subheader("Fracture Parameters")
    fracture = st.sidebar.checkbox("Include Fracture", value=False)
    
    if fracture:
        fracture_angle = st.sidebar.slider(
            "Fracture Angle (degrees)",
            min_value=-45.0,
            max_value=45.0,
            value=0.0,
            step=5.0,
            help="Angle of the fracture line"
        )
        
        fracture_width = st.sidebar.slider(
            "Fracture Width (pixels)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Width of the fracture gap"
        )

# run simulation button
run_simulation = st.sidebar.button("Run Simulation", type="primary")

# additional tools
st.sidebar.markdown("---")
st.sidebar.subheader("Additional Tools")

# Test Simulator - Display results inline
if st.sidebar.button("Run Test Simulator"):
    st.session_state.show_tests = True

# Batch Analysis - Display results inline
if st.sidebar.button("Run Batch Analysis"):
    st.session_state.show_batch = True

# main content area
if run_simulation or 'last_image' not in st.session_state:
    with st.spinner("Running simulation..."):
        # determine phantom type string
        phantom_type_str = "2d_test" if phantom_type == "2D Test Phantom" else "3d_leg"
        
        # run simulation
        phantom, xray_image = simulator.simulate_xray(
            phantom_type=phantom_type_str,
            energy=energy,
            source_distance=source_distance,
            object_distance=object_distance,
            film_distance=film_distance,
            beam_angle=beam_angle,
            fracture=fracture,
            fracture_angle=fracture_angle,
            fracture_width=fracture_width,
            noise_level=noise_level
        )
        
        # store in session state
        st.session_state.last_phantom = phantom
        st.session_state.last_image = xray_image
        st.session_state.last_params = {
            'phantom_type': phantom_type,
            'energy': energy,
            'source_distance': source_distance,
            'object_distance': object_distance,
            'film_distance': film_distance,
            'beam_angle': beam_angle,
            'noise_level': noise_level,
            'fracture': fracture,
            'fracture_angle': fracture_angle,
            'fracture_width': fracture_width
        }

# Display Test Results if requested
if 'show_tests' in st.session_state and st.session_state.show_tests:
    st.header("Test Simulator Results")
    
    with st.spinner("Running all tests..."):
        # Test 1: 2D phantom
        st.subheader("Test 1: 2D Test Phantom")
        try:
            phantom_2d, xray_2d = simulator.simulate_xray(
                phantom_type='2d_test',
                energy=60.0,
                noise_level=0.01
            )
            
            fig1 = visualizer.plot_phantom_and_xray(phantom_2d, xray_2d, "2D Test Phantom X-Ray")
            st.pyplot(fig1)
            plt.close()
            
            st.success(f"✓ 2D simulation successful - Shape: {xray_2d.shape}, Range: [{xray_2d.min():.3f}, {xray_2d.max():.3f}]")
        except Exception as e:
            st.error(f"✗ 2D simulation failed: {e}")
        
        # Test 2: 3D leg phantom with fracture
        st.subheader("Test 2: 3D Leg Phantom (With Fracture)")
        try:
            phantom_3d, xray_3d = simulator.simulate_xray(
                phantom_type='3d_leg',
                energy=60.0,
                fracture=True,
                fracture_width=2.0,
                noise_level=0.01
            )
            
            fig2 = visualizer.plot_phantom_and_xray(phantom_3d, xray_3d, 
                                                    "3D Leg Phantom X-Ray (With Fracture)")
            st.pyplot(fig2)
            plt.close()
            
            st.success(f"✓ 3D simulation successful - Shape: {xray_3d.shape}, Range: [{xray_3d.min():.3f}, {xray_3d.max():.3f}]")
        except Exception as e:
            st.error(f"✗ 3D simulation failed: {e}")
        
        # Test 3: Image metrics
        st.subheader("Test 3: Image Quality Metrics")
        try:
            metrics = ImageMetrics.get_all_metrics(xray_2d)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Contrast", f"{metrics['contrast']:.4f}")
                st.metric("Mean Intensity", f"{metrics['mean_intensity']:.4f}")
            with col2:
                st.metric("Sharpness", f"{metrics['variance_of_laplacian']:.4f}")
                st.metric("Gradient Entropy", f"{metrics['gradient_entropy']:.4f}")
            with col3:
                st.metric("Edge Strength", f"{metrics['edge_strength']:.4f}")
                st.metric("Std Intensity", f"{metrics['std_intensity']:.4f}")
            
            st.success("✓ Metrics calculated successfully")
        except Exception as e:
            st.error(f"✗ Metrics calculation failed: {e}")
        
        # Test 4: Intensity profiles
        st.subheader("Test 4: Intensity Profiles")
        try:
            fig3 = visualizer.plot_intensity_profile(xray_3d, axis=0, 
                                                     title="Horizontal Intensity Profile")
            st.pyplot(fig3)
            plt.close()
            
            fig4 = visualizer.plot_intensity_profile(xray_3d, axis=1,
                                                     title="Vertical Intensity Profile")
            st.pyplot(fig4)
            plt.close()
            
            st.success("✓ Visualization successful")
        except Exception as e:
            st.error(f"✗ Visualization failed: {e}")
        
        # Test 5: Parameter variations
        st.subheader("Test 5: Parameter Variations (Energy Sweep)")
        try:
            energies = [30, 60, 100]
            contrasts = []
            
            for e in energies:
                _, img = simulator.simulate_xray(
                    phantom_type='2d_test',
                    energy=e,
                    noise_level=0.01
                )
                contrast = ImageMetrics.calculate_contrast(img)
                contrasts.append(contrast)
            
            # Display as table
            import pandas as pd
            df = pd.DataFrame({
                'Energy (keV)': energies,
                'Contrast': contrasts
            })
            st.dataframe(df)
            
            st.success("✓ Parameter variations working correctly")
        except Exception as e:
            st.error(f"✗ Parameter variation test failed: {e}")
        
        # Test 6: Attenuation coefficients
        st.subheader("Test 6: Attenuation Coefficient Lookup")
        try:
            materials = ['air', 'soft_tissue', 'bone']
            test_energy = 60.0
            
            data = []
            for material in materials:
                mu = simulator.get_attenuation_coefficient(material, test_energy)
                data.append({'Material': material.replace('_', ' ').title(), 
                           f'μ @ {test_energy} keV (cm⁻¹)': f"{mu:.5f}"})
            
            import pandas as pd
            df = pd.DataFrame(data)
            st.dataframe(df)
            
            st.success("✓ Attenuation coefficients retrieved successfully")
        except Exception as e:
            st.error(f"✗ Attenuation coefficient test failed: {e}")
    
    st.success("🎉 ALL TESTS PASSED!")
    
    # Reset flag
    if st.button("Close Test Results"):
        st.session_state.show_tests = False
        st.rerun()

# Display Batch Analysis Results if requested
if 'show_batch' in st.session_state and st.session_state.show_batch:
    st.header("Batch Analysis Results")
    
    with st.spinner("Running comprehensive parameter sweeps..."):
        
        # Sample Images
        st.subheader("Sample Images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**2D Test Phantom**")
            phantom_2d, xray_2d = simulator.simulate_xray(
                phantom_type="2d_test",
                energy=60.0,
                noise_level=0.01
            )
            fig1 = visualizer.plot_phantom_and_xray(phantom_2d, xray_2d, "2D Test Phantom X-Ray")
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.write("**3D Leg Phantom (With Fracture)**")
            phantom_3d_frac, xray_3d_frac = simulator.simulate_xray(
                phantom_type="3d_leg",
                energy=60.0,
                fracture=True,
                fracture_angle=15.0,
                fracture_width=2.5,
                noise_level=0.01
            )
            fig2 = visualizer.plot_phantom_and_xray(phantom_3d_frac, xray_3d_frac,
                                                    "3D Leg Phantom X-Ray (With Fracture)")
            st.pyplot(fig2)
            plt.close()
        
        # Energy Sweep
        st.subheader("1. Energy Sweep Analysis")
        energies = [20, 40, 60, 80, 100, 120, 150]
        
        images = []
        metrics_list = []
        
        for e in energies:
            _, img = simulator.simulate_xray(
                phantom_type='2d_test',
                energy=e,
                noise_level=0.01
            )
            images.append(img)
            metrics_list.append(ImageMetrics.get_all_metrics(img))
        
        fig3 = visualizer.plot_energy_sweep(energies, images, metrics_list)
        st.pyplot(fig3)
        plt.close()
        
        # Show metrics table
        import pandas as pd
        df = pd.DataFrame(metrics_list)
        df.insert(0, 'Energy (keV)', energies)
        st.dataframe(df.round(4))
        
        # Distance Sweep
        st.subheader("2. Distance Sweep Analysis")
        source_distances = [80, 100, 120, 150, 180]
        
        dist_images = []
        dist_labels = []
        dist_metrics = []
        
        for sd in source_distances:
            object_dist = 50.0
            film_dist = sd + 50.0
            
            _, img = simulator.simulate_xray(
                phantom_type='2d_test',
                energy=60.0,
                source_distance=sd,
                object_distance=object_dist,
                film_distance=film_dist,
                noise_level=0.01
            )
            
            mag = film_dist / object_dist
            dist_images.append(img)
            dist_labels.append(f"SD={sd}cm (M={mag:.2f}x)")
            dist_metrics.append(ImageMetrics.get_all_metrics(img))
        
        fig4 = visualizer.plot_parameter_comparison(dist_images, dist_labels,
                                                    "Distance Parameter Sweep")
        st.pyplot(fig4)
        plt.close()
        
        # Metrics comparison
        metrics_dict = {
            'labels': dist_labels,
            'contrast': [m['contrast'] for m in dist_metrics],
            'sharpness': [m['variance_of_laplacian'] for m in dist_metrics],
            'edge_strength': [m['edge_strength'] for m in dist_metrics]
        }
        fig5 = visualizer.plot_metrics_comparison(metrics_dict, "Distance Sweep Metrics")
        st.pyplot(fig5)
        plt.close()
        
        # Angle Sweep
        st.subheader("3. Beam Angle Sweep Analysis")
        angles = [-30, -15, 0, 15, 30]
        
        angle_images = []
        angle_labels = []
        angle_metrics = []
        
        for angle in angles:
            _, img = simulator.simulate_xray(
                phantom_type='2d_test',
                energy=60.0,
                beam_angle=angle,
                noise_level=0.01
            )
            
            angle_images.append(img)
            angle_labels.append(f"{angle}°")
            angle_metrics.append(ImageMetrics.get_all_metrics(img))
        
        fig6 = visualizer.plot_parameter_comparison(angle_images, angle_labels,
                                                    "Beam Angle Parameter Sweep")
        st.pyplot(fig6)
        plt.close()
        
        # Angle metrics
        angle_metrics_dict = {
            'labels': angle_labels,
            'contrast': [m['contrast'] for m in angle_metrics],
            'sharpness': [m['variance_of_laplacian'] for m in angle_metrics],
            'edge_strength': [m['edge_strength'] for m in angle_metrics]
        }
        fig7 = visualizer.plot_metrics_comparison(angle_metrics_dict, "Angle Sweep Metrics")
        st.pyplot(fig7)
        plt.close()
        
        # Fracture Analysis
        st.subheader("4. Fracture Parameter Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fracture Width Comparison**")
            fracture_widths = [1.0, 2.0, 3.0, 5.0]
            width_images = []
            width_labels = []
            
            for width in fracture_widths:
                _, img = simulator.simulate_xray(
                    phantom_type="3d_leg",
                    energy=60.0,
                    fracture=True,
                    fracture_width=width,
                    fracture_angle=0.0,
                    noise_level=0.01
                )
                width_images.append(img)
                width_labels.append(f"Width={width}px")
            
            fig8 = visualizer.plot_parameter_comparison(width_images, width_labels,
                                                        "Fracture Width Comparison")
            st.pyplot(fig8)
            plt.close()
        
        with col2:
            st.write("**Fracture Angle Comparison**")
            fracture_angles = [0, 15, 30, 45]
            fangle_images = []
            fangle_labels = []
            
            for fa in fracture_angles:
                _, img = simulator.simulate_xray(
                    phantom_type="3d_leg",
                    energy=60.0,
                    fracture=True,
                    fracture_width=2.0,
                    fracture_angle=fa,
                    noise_level=0.01
                )
                fangle_images.append(img)
                fangle_labels.append(f"Angle={fa}°")
            
            fig9 = visualizer.plot_parameter_comparison(fangle_images, fangle_labels,
                                                        "Fracture Angle Comparison")
            st.pyplot(fig9)
            plt.close()
    
    st.success("Complete batch analysis finished!")
    
    # Reset flag
    if st.button("Close Batch Results"):
        st.session_state.show_batch = False
        st.rerun()

# display results if available
if 'last_image' in st.session_state and ('show_tests' not in st.session_state or not st.session_state.show_tests) and ('show_batch' not in st.session_state or not st.session_state.show_batch):
    phantom = st.session_state.last_phantom
    xray_image = st.session_state.last_image
    params = st.session_state.last_params
    
    # create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Intensity Profiles", "Metrics", "Information"])
    
    with tab1:
        st.header("Simulation Results")
        
        # display phantom and x-ray side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Phantom (Material Map)")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im1 = ax1.imshow(phantom, cmap='viridis', interpolation='nearest')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, label='Material Type')
            st.pyplot(fig1)
            plt.close()
            
        with col2:
            st.subheader("X-Ray Image")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            im2 = ax2.imshow(xray_image, cmap='gray', interpolation='bilinear')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Intensity')
            st.pyplot(fig2)
            plt.close()
        
        # display current parameters
        st.subheader("Current Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Energy", f"{params['energy']:.0f} keV")
            st.metric("Source Distance", f"{params['source_distance']:.0f} cm")
            st.metric("Beam Angle", f"{params['beam_angle']:.0f}°")
        
        with col2:
            st.metric("Object Distance", f"{params['object_distance']:.0f} cm")
            st.metric("Film Distance", f"{params['film_distance']:.0f} cm")
            magnification = params['film_distance'] / params['object_distance']
            st.metric("Magnification", f"{magnification:.2f}x")
        
        with col3:
            st.metric("Noise Level", f"{params['noise_level']:.3f}")
            if params['fracture']:
                st.metric("Fracture Angle", f"{params['fracture_angle']:.0f}°")
                st.metric("Fracture Width", f"{params['fracture_width']:.1f} px")
    
    with tab2:
        st.header("Intensity Profiles")
        
        # horizontal profile
        st.subheader("Horizontal Profile (Center)")
        h_position = xray_image.shape[0] // 2
        h_profile = xray_image[h_position, :]
        
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 4))
        
        ax3a.imshow(xray_image, cmap='gray')
        ax3a.axhline(y=h_position, color='r', linestyle='--', linewidth=2)
        ax3a.set_title('Image with Profile Line')
        ax3a.axis('off')
        
        ax3b.plot(h_profile, linewidth=2)
        ax3b.set_xlabel('Horizontal Position (pixels)')
        ax3b.set_ylabel('Intensity')
        ax3b.set_title('Horizontal Intensity Profile')
        ax3b.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        plt.close()
        
        # vertical profile
        st.subheader("Vertical Profile (Center)")
        v_position = xray_image.shape[1] // 2
        v_profile = xray_image[:, v_position]
        
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 4))
        
        ax4a.imshow(xray_image, cmap='gray')
        ax4a.axvline(x=v_position, color='r', linestyle='--', linewidth=2)
        ax4a.set_title('Image with Profile Line')
        ax4a.axis('off')
        
        ax4b.plot(v_profile, linewidth=2)
        ax4b.set_xlabel('Vertical Position (pixels)')
        ax4b.set_ylabel('Intensity')
        ax4b.set_title('Vertical Intensity Profile')
        ax4b.grid(True, alpha=0.3)
        
        st.pyplot(fig4)
        plt.close()
    
    with tab3:
        st.header("Image Quality Metrics")
        
        # calculate metrics
        metrics = ImageMetrics.get_all_metrics(xray_image)
        
        # display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Contrast", f"{metrics['contrast']:.4f}")
            st.metric("Mean Intensity", f"{metrics['mean_intensity']:.4f}")
        
        with col2:
            st.metric("Variance of Laplacian", f"{metrics['variance_of_laplacian']:.4f}")
            st.metric("Gradient Entropy", f"{metrics['gradient_entropy']:.4f}")
        
        with col3:
            st.metric("Edge Strength", f"{metrics['edge_strength']:.4f}")
            st.metric("Std Intensity", f"{metrics['std_intensity']:.4f}")
        
        # metrics explanation
        st.subheader("Metrics Explanation")
        st.markdown("""
        - **Contrast**: Measure of intensity difference between regions (higher = better differentiation)
        - **Variance of Laplacian**: Measure of image sharpness (higher = sharper edges)
        - **Gradient Entropy**: Measure of edge information content (higher = more detail)
        - **Edge Strength**: Average magnitude of edges in the image
        - **Mean/Std Intensity**: Basic intensity statistics
        """)
        
        # fracture-specific metrics if applicable
        if params['fracture'] and params['phantom_type'] == "3D Leg Phantom":
            st.subheader("Fracture Visibility Metrics")
            
            # define fracture region (center of image)
            h, w = xray_image.shape
            margin = 20
            fracture_region = (h//2 - margin, h//2 + margin, w//2 - margin, w//2 + margin)
            
            fracture_metrics = ImageMetrics.calculate_fracture_visibility(
                xray_image, fracture_region
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fracture Contrast", f"{fracture_metrics['fracture_contrast']:.4f}")
                st.metric("Fracture Sharpness", f"{fracture_metrics['fracture_sharpness']:.4f}")
            
            with col2:
                st.metric("Fracture Edge Strength", f"{fracture_metrics['fracture_edge_strength']:.4f}")
                intensity_diff = abs(fracture_metrics['fracture_mean_intensity'] - 
                                   fracture_metrics['surrounding_mean_intensity'])
                st.metric("Intensity Difference", f"{intensity_diff:.4f}")
    
    with tab4:
        st.header("System Information")
        
        st.subheader("About This Simulation")
        st.markdown("""
        This virtual x-ray simulator models x-ray imaging using the **Beer-Lambert Law**:
        
        $$I = I_0 \\cdot e^{-\\mu d}$$
        
        Where:
        - $I$ = transmitted intensity
        - $I_0$ = initial intensity
        - $\\mu$ = linear attenuation coefficient
        - $d$ = material thickness
        
        The simulator accounts for:
        - Material-specific attenuation at different energies
        - Geometric magnification effects
        - Beam angle variations
        - Quantum noise
        - Fracture modeling (for 3D leg phantom)
        """)
        
        st.subheader("Parameter Effects")
        st.markdown("""
        **Energy**: 
        - Higher energy → more penetration, less contrast
        - Lower energy → less penetration, higher contrast
        
        **Distances**:
        - Magnification = Film Distance / Object Distance
        - Larger magnification → bigger but potentially blurrier image
        
        **Beam Angle**:
        - Non-zero angles distort the projected image
        - Useful for visualizing 3D structure from different perspectives
        
        **Fracture Parameters**:
        - Fracture width affects visibility
        - Fracture angle changes appearance in projection
        """)
        
        st.subheader("Attenuation Coefficients")
        st.markdown(f"""
        Current energy: **{params['energy']:.0f} keV**
        
        - Air: {simulator.get_attenuation_coefficient('air', params['energy']):.5f} cm⁻¹
        - Soft Tissue: {simulator.get_attenuation_coefficient('soft_tissue', params['energy']):.4f} cm⁻¹
        - Bone: {simulator.get_attenuation_coefficient('bone', params['energy']):.4f} cm⁻¹
        """)

else:
    if 'show_tests' not in st.session_state or not st.session_state.show_tests:
        if 'show_batch' not in st.session_state or not st.session_state.show_batch:
            st.info("Click 'Run Simulation' in the sidebar to generate an x-ray image")

# footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Virtual X-Ray Simulator**  
Medical Imaging Project 2025  
Authors: Amber Kaul & Derek Herrera
""")