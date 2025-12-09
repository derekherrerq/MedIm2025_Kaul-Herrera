"""
visualization utilities for x-ray simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple


class XRayVisualizer:
    """
    handles all visualization for x-ray simulations.
    """
    
    @staticmethod
    def plot_phantom_and_xray(phantom: np.ndarray, xray_image: np.ndarray,
                             title: str = "X-Ray Simulation") -> Figure:
        """
        plot phantom and resulting x-ray image side by side.
        
        args:
            phantom: phantom array
            xray_image: x-ray intensity image
            title: plot title
            
        returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # plot phantom
        im1 = axes[0].imshow(phantom, cmap='viridis', interpolation='nearest')
        axes[0].set_title('Phantom (Material Map)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Material Type')
        
        # add legend for materials
        materials = {0: 'Air', 1: 'Soft Tissue', 2: 'Bone'}
        patches = [mpatches.Patch(color=plt.cm.viridis(i/2), label=materials.get(i, f'Material {i}'))
                  for i in np.unique(phantom)]
        axes[0].legend(handles=patches, loc='upper right', fontsize=8)
        
        # plot x-ray image
        im2 = axes[1].imshow(xray_image, cmap='gray', interpolation='bilinear')
        axes[1].set_title('X-Ray Image (Intensity)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Intensity')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_intensity_profile(image: np.ndarray, 
                              axis: int = 0,
                              position: int = None,
                              title: str = "Intensity Profile") -> Figure:
        """
        plot intensity profile through image.
        
        args:
            image: x-ray image
            axis: axis for profile (0=horizontal, 1=vertical)
            position: position along perpendicular axis
            title: plot title
            
        returns:
            matplotlib figure
        """
        if position is None:
            position = image.shape[1 - axis] // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # show image with profile line
        axes[0].imshow(image, cmap='gray')
        
        if axis == 0:
            # horizontal profile
            axes[0].axhline(y=position, color='r', linestyle='--', linewidth=2, label='Profile Line')
            profile = image[position, :]
            x_axis = np.arange(len(profile))
            xlabel = 'Horizontal Position (pixels)'
        else:
            # vertical profile
            axes[0].axvline(x=position, color='r', linestyle='--', linewidth=2, label='Profile Line')
            profile = image[:, position]
            x_axis = np.arange(len(profile))
            xlabel = 'Vertical Position (pixels)'
        
        axes[0].set_title('X-Ray Image with Profile Line')
        axes[0].legend()
        axes[0].axis('off')
        
        # plot profile
        axes[1].plot(x_axis, profile, linewidth=2)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('Intensity')
        axes[1].set_title('Intensity Profile')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_2d_profile_heatmap(image: np.ndarray, 
                               num_profiles: int = 10,
                               axis: int = 0,
                               title: str = "2D Profile View") -> Figure:
        """
        plot multiple profiles as a heatmap.
        
        args:
            image: x-ray image
            num_profiles: number of profiles to extract
            axis: axis for profiles
            title: plot title
            
        returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        profiles = []
        positions = np.linspace(0, image.shape[1-axis]-1, num_profiles, dtype=int)
        
        for pos in positions:
            if axis == 0:
                profile = image[pos, :]
            else:
                profile = image[:, pos]
            profiles.append(profile)
        
        profiles_array = np.array(profiles)
        
        im = ax.imshow(profiles_array, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.set_xlabel('Position along profile')
        ax.set_ylabel('Profile number')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_parameter_comparison(images: List[np.ndarray],
                                 labels: List[str],
                                 title: str = "Parameter Comparison") -> Figure:
        """
        plot multiple images for parameter comparison.
        
        args:
            images: list of x-ray images
            labels: list of labels for each image
            title: overall title
            
        returns:
            matplotlib figure
        """
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            ax = axes[idx] if n_images > 1 else axes[0]
            im = ax.imshow(img, cmap='gray', interpolation='bilinear')
            ax.set_title(label)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, List],
                               title: str = "Metrics Comparison") -> Figure:
        """
        plot bar charts comparing metrics across different settings.
        
        args:
            metrics_dict: dictionary with metric names and values
            title: plot title
            
        returns:
            matplotlib figure
        """
        labels = metrics_dict.get('labels', [])
        
        # metrics to plot
        metric_names = ['contrast', 'sharpness', 'edge_strength']
        available_metrics = [m for m in metric_names if m in metrics_dict]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            values = metrics_dict[metric]
            x_pos = np.arange(len(labels))
            
            axes[idx].bar(x_pos, values, alpha=0.7, color='steelblue')
            axes[idx].set_xlabel('Configuration')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(labels, rotation=45, ha='right')
            axes[idx].grid(axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_metrics_table(metrics_dict: Dict[str, float]) -> Figure:
        """
        create a table visualization of metrics.
        
        args:
            metrics_dict: dictionary of metric names and values
            
        returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, len(metrics_dict) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        # prepare table data
        table_data = [[key.replace('_', ' ').title(), f'{value:.4f}'] 
                     for key, value in metrics_dict.items()]
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # alternate row colors
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Image Quality Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_energy_sweep(energies: List[float],
                         images: List[np.ndarray],
                         metrics: List[Dict[str, float]],
                         title: str = "Energy Sweep Analysis") -> Figure:
        """
        plot results of energy parameter sweep.
        
        args:
            energies: list of energy values
            images: list of corresponding images
            metrics: list of metric dictionaries
            title: plot title
            
        returns:
            matplotlib figure
        """
        n_images = min(4, len(images))
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, n_images, hspace=0.3, wspace=0.3)
        
        # plot sample images
        for i in range(n_images):
            idx = i * (len(images) // n_images)
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(f'{energies[idx]} keV')
            ax.axis('off')
        
        # extract metrics
        contrast_values = [m.get('contrast', 0) for m in metrics]
        sharpness_values = [m.get('variance_of_laplacian', 0) for m in metrics]
        edge_values = [m.get('edge_strength', 0) for m in metrics]
        
        # plot contrast vs energy
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.plot(energies, contrast_values, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Contrast')
        ax1.set_title('Contrast vs Energy')
        ax1.grid(True, alpha=0.3)
        
        # plot sharpness vs energy
        ax2 = fig.add_subplot(gs[1, 2:])
        ax2.plot(energies, sharpness_values, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('Sharpness (Var. of Laplacian)')
        ax2.set_title('Sharpness vs Energy')
        ax2.grid(True, alpha=0.3)
        
        # plot edge strength vs energy
        ax3 = fig.add_subplot(gs[2, 1:3])
        ax3.plot(energies, edge_values, marker='^', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Energy (keV)')
        ax3.set_ylabel('Edge Strength')
        ax3.set_title('Edge Strength vs Energy')
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        return fig
    
    @staticmethod
    def save_figure(fig: Figure, filename: str, dpi: int = 300):
        """
        save figure to file.
        
        args:
            fig: matplotlib figure
            filename: output filename
            dpi: resolution
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)