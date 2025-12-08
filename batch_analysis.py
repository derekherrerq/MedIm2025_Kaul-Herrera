"""
batch analysis script - performs parameter sweeps and generates comprehensive analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
from xray_simulator import XRaySimulator
from image_metrics import ImageMetrics
from visualization import XRayVisualizer
import os
from typing import List, Dict
import pandas as pd


class BatchAnalysis:
    """
    performs batch analysis of x-ray simulations with parameter sweeps.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        initialize batch analysis.
        
        args:
            output_dir: directory to save results
        """
        self.simulator = XRaySimulator(image_size=(512, 512))
        self.visualizer = XRayVisualizer()
        self.output_dir = output_dir
        
        # make output folder
        os.makedirs(output_dir, exist_ok=True)
        
    def energy_sweep(self, phantom_type: str = "2d_test",
                    energies: List[float] = None,
                    save_results: bool = True) -> Dict:
        """
        do energy parameter sweep.
        
        args:
            phantom_type: type of phantom to use
            energies: list of energies to test (kev)
            save_results: whether to save results to disk
            
        returns:
            dictionary with results
        """
        if energies is None:
            energies = [20, 40, 60, 80, 100, 120, 150]
        
        print(f"Running energy sweep with {len(energies)} values...")
        
        images = []
        metrics_list = []
        
        for energy in energies:
            print(f"  Processing energy: {energy} keV")
            _, xray_image = self.simulator.simulate_xray(
                phantom_type=phantom_type,
                energy=energy,
                noise_level=0.01
            )
            
            images.append(xray_image)
            metrics = ImageMetrics.get_all_metrics(xray_image)
            metrics_list.append(metrics)
        
        # make visualization
        fig = self.visualizer.plot_energy_sweep(energies, images, metrics_list)
        
        if save_results:
            output_path = os.path.join(self.output_dir, "energy_sweep_analysis.png")
            self.visualizer.save_figure(fig, output_path)
            print(f"Saved energy sweep analysis to {output_path}")
            
            # save metrics to csv
            df = pd.DataFrame(metrics_list)
            df['energy_keV'] = energies
            csv_path = os.path.join(self.output_dir, "energy_sweep_metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved metrics to {csv_path}")
        
        return {
            'energies': energies,
            'images': images,
            'metrics': metrics_list
        }
    
    def distance_sweep(self, phantom_type: str = "2d_test",
                      source_distances: List[float] = None,
                      save_results: bool = True) -> Dict:
        """
        do distance parameter sweep.
        
        args:
            phantom_type: type of phantom to use
            source_distances: list of source distances to test (cm)
            save_results: whether to save results to disk
            
        returns:
            dictionary with results
        """
        if source_distances is None:
            source_distances = [80, 100, 120, 150, 180]
        
        print(f"Running distance sweep with {len(source_distances)} values...")
        
        images = []
        metrics_list = []
        magnifications = []
        
        for sd in source_distances:
            print(f"  Processing source distance: {sd} cm")
            object_dist = 50.0
            film_dist = sd + 50.0
            
            _, xray_image = self.simulator.simulate_xray(
                phantom_type=phantom_type,
                energy=60.0,
                source_distance=sd,
                object_distance=object_dist,
                film_distance=film_dist,
                noise_level=0.01
            )
            
            images.append(xray_image)
            metrics = ImageMetrics.get_all_metrics(xray_image)
            metrics_list.append(metrics)
            magnifications.append(film_dist / object_dist)
        
        # make comparison visualization
        labels = [f"SD={sd}cm (M={m:.2f}x)" for sd, m in zip(source_distances, magnifications)]
        fig = self.visualizer.plot_parameter_comparison(images, labels, 
                                                        "Distance Parameter Sweep")
        
        if save_results:
            output_path = os.path.join(self.output_dir, "distance_sweep_comparison.png")
            self.visualizer.save_figure(fig, output_path)
            print(f"Saved distance sweep to {output_path}")
            
            # plot metrics
            metrics_dict = {
                'labels': labels,
                'contrast': [m['contrast'] for m in metrics_list],
                'sharpness': [m['variance_of_laplacian'] for m in metrics_list],
                'edge_strength': [m['edge_strength'] for m in metrics_list]
            }
            fig2 = self.visualizer.plot_metrics_comparison(metrics_dict, 
                                                          "Distance Sweep Metrics")
            output_path2 = os.path.join(self.output_dir, "distance_sweep_metrics.png")
            self.visualizer.save_figure(fig2, output_path2)
            
            # save to csv
            df = pd.DataFrame(metrics_list)
            df['source_distance_cm'] = source_distances
            df['magnification'] = magnifications
            csv_path = os.path.join(self.output_dir, "distance_sweep_metrics.csv")
            df.to_csv(csv_path, index=False)
        
        return {
            'source_distances': source_distances,
            'images': images,
            'metrics': metrics_list,
            'magnifications': magnifications
        }
    
    def angle_sweep(self, phantom_type: str = "2d_test",
                   angles: List[float] = None,
                   save_results: bool = True) -> Dict:
        """
        do beam angle parameter sweep.
        
        args:
            phantom_type: type of phantom to use
            angles: list of beam angles to test (degrees)
            save_results: whether to save results to disk
            
        returns:
            dictionary with results
        """
        if angles is None:
            angles = [-30, -15, 0, 15, 30]
        
        print(f"Running angle sweep with {len(angles)} values...")
        
        images = []
        metrics_list = []
        
        for angle in angles:
            print(f"  Processing angle: {angle}°")
            _, xray_image = self.simulator.simulate_xray(
                phantom_type=phantom_type,
                energy=60.0,
                beam_angle=angle,
                noise_level=0.01
            )
            
            images.append(xray_image)
            metrics = ImageMetrics.get_all_metrics(xray_image)
            metrics_list.append(metrics)
        
        # make comparison visualization
        labels = [f"{angle}°" for angle in angles]
        fig = self.visualizer.plot_parameter_comparison(images, labels, 
                                                        "Beam Angle Parameter Sweep")
        
        if save_results:
            output_path = os.path.join(self.output_dir, "angle_sweep_comparison.png")
            self.visualizer.save_figure(fig, output_path)
            print(f"Saved angle sweep to {output_path}")
            
            # plot metrics
            metrics_dict = {
                'labels': labels,
                'contrast': [m['contrast'] for m in metrics_list],
                'sharpness': [m['variance_of_laplacian'] for m in metrics_list],
                'edge_strength': [m['edge_strength'] for m in metrics_list]
            }
            fig2 = self.visualizer.plot_metrics_comparison(metrics_dict, 
                                                          "Angle Sweep Metrics")
            output_path2 = os.path.join(self.output_dir, "angle_sweep_metrics.png")
            self.visualizer.save_figure(fig2, output_path2)
            
            # save to csv
            df = pd.DataFrame(metrics_list)
            df['beam_angle_deg'] = angles
            csv_path = os.path.join(self.output_dir, "angle_sweep_metrics.csv")
            df.to_csv(csv_path, index=False)
        
        return {
            'angles': angles,
            'images': images,
            'metrics': metrics_list
        }
    
    def fracture_analysis(self, fracture_widths: List[float] = None,
                         fracture_angles: List[float] = None,
                         save_results: bool = True) -> Dict:
        """
        analyze fracture visibility with different parameters.
        
        args:
            fracture_widths: list of fracture widths to test (pixels)
            fracture_angles: list of fracture angles to test (degrees)
            save_results: whether to save results to disk
            
        returns:
            dictionary with results
        """
        if fracture_widths is None:
            fracture_widths = [1.0, 2.0, 3.0, 5.0]
        
        if fracture_angles is None:
            fracture_angles = [0, 15, 30, 45]
        
        print("Running fracture analysis...")
        
        # test fracture widths
        print(f"  Testing {len(fracture_widths)} fracture widths...")
        width_images = []
        width_metrics = []
        
        for width in fracture_widths:
            print(f"    Width: {width} px")
            _, xray_image = self.simulator.simulate_xray(
                phantom_type="3d_leg",
                energy=60.0,
                fracture=True,
                fracture_width=width,
                fracture_angle=0.0,
                noise_level=0.01
            )
            
            width_images.append(xray_image)
            
            # calculate fracture-specific metrics
            h, w = xray_image.shape
            margin = 20
            fracture_region = (h//2 - margin, h//2 + margin, w//2 - margin, w//2 + margin)
            metrics = ImageMetrics.calculate_fracture_visibility(xray_image, fracture_region)
            width_metrics.append(metrics)
        
        # test fracture angles
        print(f"  Testing {len(fracture_angles)} fracture angles...")
        angle_images = []
        angle_metrics = []
        
        for angle in fracture_angles:
            print(f"    Angle: {angle}°")
            _, xray_image = self.simulator.simulate_xray(
                phantom_type="3d_leg",
                energy=60.0,
                fracture=True,
                fracture_width=2.0,
                fracture_angle=angle,
                noise_level=0.01
            )
            
            angle_images.append(xray_image)
            
            h, w = xray_image.shape
            margin = 20
            fracture_region = (h//2 - margin, h//2 + margin, w//2 - margin, w//2 + margin)
            metrics = ImageMetrics.calculate_fracture_visibility(xray_image, fracture_region)
            angle_metrics.append(metrics)
        
        if save_results:
            # visualize width comparison
            width_labels = [f"Width={w}px" for w in fracture_widths]
            fig1 = self.visualizer.plot_parameter_comparison(width_images, width_labels,
                                                            "Fracture Width Comparison")
            output_path1 = os.path.join(self.output_dir, "fracture_width_comparison.png")
            self.visualizer.save_figure(fig1, output_path1)
            
            # visualize angle comparison
            angle_labels = [f"Angle={a}°" for a in fracture_angles]
            fig2 = self.visualizer.plot_parameter_comparison(angle_images, angle_labels,
                                                            "Fracture Angle Comparison")
            output_path2 = os.path.join(self.output_dir, "fracture_angle_comparison.png")
            self.visualizer.save_figure(fig2, output_path2)
            
            print(f"Saved fracture analysis to {self.output_dir}")
            
            # save metrics to csv
            df_width = pd.DataFrame(width_metrics)
            df_width['fracture_width_px'] = fracture_widths
            csv_path1 = os.path.join(self.output_dir, "fracture_width_metrics.csv")
            df_width.to_csv(csv_path1, index=False)
            
            df_angle = pd.DataFrame(angle_metrics)
            df_angle['fracture_angle_deg'] = fracture_angles
            csv_path2 = os.path.join(self.output_dir, "fracture_angle_metrics.csv")
            df_angle.to_csv(csv_path2, index=False)
        
        return {
            'fracture_widths': fracture_widths,
            'width_images': width_images,
            'width_metrics': width_metrics,
            'fracture_angles': fracture_angles,
            'angle_images': angle_images,
            'angle_metrics': angle_metrics
        }
    
    def generate_sample_images(self, save_results: bool = True):
        """
        generate sample x-ray images for demonstration.
        
        args:
            save_results: whether to save results to disk
        """
        print("Generating sample images...")
        
        # 2d test phantom
        print("  Generating 2D test phantom...")
        phantom_2d, xray_2d = self.simulator.simulate_xray(
            phantom_type="2d_test",
            energy=60.0,
            noise_level=0.01
        )
        
        fig1 = self.visualizer.plot_phantom_and_xray(phantom_2d, xray_2d,
                                                     "2D Test Phantom X-Ray")
        
        # 3d leg phantom without fracture
        print("  Generating 3D leg phantom (no fracture)...")
        phantom_3d, xray_3d = self.simulator.simulate_xray(
            phantom_type="3d_leg",
            energy=60.0,
            fracture=False,
            noise_level=0.01
        )
        
        fig2 = self.visualizer.plot_phantom_and_xray(phantom_3d, xray_3d,
                                                     "3D Leg Phantom X-Ray (No Fracture)")
        
        # 3d leg phantom with fracture
        print("  Generating 3D leg phantom (with fracture)...")
        phantom_3d_frac, xray_3d_frac = self.simulator.simulate_xray(
            phantom_type="3d_leg",
            energy=60.0,
            fracture=True,
            fracture_angle=15.0,
            fracture_width=2.5,
            noise_level=0.01
        )
        
        fig3 = self.visualizer.plot_phantom_and_xray(phantom_3d_frac, xray_3d_frac,
                                                     "3D Leg Phantom X-Ray (With Fracture)")
        
        # generate intensity profiles
        fig4 = self.visualizer.plot_intensity_profile(xray_3d_frac, axis=0,
                                                      title="Horizontal Intensity Profile")
        
        fig5 = self.visualizer.plot_intensity_profile(xray_3d_frac, axis=1,
                                                      title="Vertical Intensity Profile")
        
        if save_results:
            self.visualizer.save_figure(fig1, os.path.join(self.output_dir, "sample_2d_test.png"))
            self.visualizer.save_figure(fig2, os.path.join(self.output_dir, "sample_3d_leg_no_fracture.png"))
            self.visualizer.save_figure(fig3, os.path.join(self.output_dir, "sample_3d_leg_with_fracture.png"))
            self.visualizer.save_figure(fig4, os.path.join(self.output_dir, "sample_horizontal_profile.png"))
            self.visualizer.save_figure(fig5, os.path.join(self.output_dir, "sample_vertical_profile.png"))
            print(f"Saved sample images to {self.output_dir}")
    
    def run_full_analysis(self):
        """
        run complete analysis pipeline.
        """
        print("="*60)
        print("Running Full Virtual X-Ray Analysis")
        print("="*60)
        
        # generate sample images
        self.generate_sample_images()
        
        # energy sweep
        print("\n" + "="*60)
        self.energy_sweep(phantom_type="2d_test")
        
        # distance sweep
        print("\n" + "="*60)
        self.distance_sweep(phantom_type="2d_test")
        
        # angle sweep
        print("\n" + "="*60)
        self.angle_sweep(phantom_type="2d_test")
        
        # fracture analysis
        print("\n" + "="*60)
        self.fracture_analysis()
        
        print("\n" + "="*60)
        print(f"Analysis complete! Results saved to '{self.output_dir}' directory")
        print("="*60)


if __name__ == "__main__":
    # run batch analysis
    analysis = BatchAnalysis(output_dir="results")
    analysis.run_full_analysis()