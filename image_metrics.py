"""
image quality metrics for x-ray analysis
implements contrast, mse, ssim, sharpness, and edge detection metrics
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.ndimage import laplace, sobel
from scipy.stats import entropy
from typing import Dict, Tuple
class ImageMetrics:
    """
    calculate various image quality metrics for x-ray images.
    """
    @staticmethod
    def calculate_contrast(image: np.ndarray, region1_mask: np.ndarray = None,
                          region2_mask: np.ndarray = None) -> float:
        """
        calculate contrast between two regions or overall image contrast.
        args:
            image: input image
            region1_mask: binary mask for region 1 (optional)
            region2_mask: binary mask for region 2 (optional)
        returns:
            contrast value (michelson contrast or weber contrast)
        """
        if region1_mask is not None and region2_mask is not None:
            # calculate contrast between two specific regions
            intensity1 = np.mean(image[region1_mask])
            intensity2 = np.mean(image[region2_mask])
            # michelson contrast
            if intensity1 + intensity2 != 0:
                contrast = np.abs(intensity1 - intensity2) / (intensity1 + intensity2)
            else:
                contrast = 0.0
        else:
            # calculate overall image contrast (rms contrast)
            mean_intensity = np.mean(image)
            contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        return float(contrast)
    @staticmethod
    def calculate_snr(image: np.ndarray, signal_mask: np.ndarray = None,
                     noise_region: np.ndarray = None) -> float:
        """
        calculate signal-to-noise ratio.
        args:
            image: input image
            signal_mask: mask for signal region
            noise_region: region to estimate noise from
        returns:
            snr value
        """
        if signal_mask is not None:
            signal_mean = np.mean(image[signal_mask])
        else:
            signal_mean = np.mean(image)
        if noise_region is not None:
            noise_std = np.std(image[noise_region])
        else:
            # estimate noise from image
            noise_std = np.std(image)
        if noise_std == 0:
            return float('inf')
        snr = signal_mean / noise_std
        return float(snr)
    @staticmethod
    def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        calculate mean squared error between two images.
        args:
            image1: first image
            image2: second image
        returns:
            mse value
        """
        return float(mse(image1, image2))
    @staticmethod
    def calculate_ssim(image1: np.ndarray, image2: np.ndarray,
                      data_range: float = None) -> float:
        """
        calculate structural similarity index.
        args:
            image1: first image
            image2: second image
            data_range: data range of images
        returns:
            ssim value
        """
        if data_range is None:
            data_range = max(image1.max(), image2.max()) - min(image1.min(), image2.min())
        return float(ssim(image1, image2, data_range=data_range))
    @staticmethod
    def calculate_variance_of_laplacian(image: np.ndarray) -> float:
        """
        calculate variance of laplacian as a sharpness metric.
        higher values indicate sharper images.
        args:
            image: input image
        returns:
            variance of laplacian
        """
        laplacian = laplace(image)
        variance = np.var(laplacian)
        return float(variance)
    @staticmethod
    def calculate_gradient_entropy(image: np.ndarray) -> float:
        """
        calculate entropy of gradient magnitudes.
        higher values indicate more edge information.
        args:
            image: input image
        returns:
            gradient entropy value
        """
        # calculate gradients
        gx = sobel(image, axis=1)
        gy = sobel(image, axis=0)
        # gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        # normalize to create probability distribution
        hist, _ = np.histogram(gradient_magnitude, bins=256, range=(0, gradient_magnitude.max()))
        hist = hist / hist.sum()
        # calculate entropy
        hist = hist[hist > 0]  # remove zeros for log calculation
        ent = entropy(hist)
        return float(ent)
    @staticmethod
    def calculate_edge_strength(image: np.ndarray) -> float:
        """
        calculate average edge strength using sobel operator.
        args:
            image: input image
        returns:
            mean edge strength
        """
        gx = sobel(image, axis=1)
        gy = sobel(image, axis=0)
        edge_magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(edge_magnitude))
    @staticmethod
    def calculate_fracture_visibility(image: np.ndarray, 
                                     fracture_region: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        calculate metrics specific to fracture visibility.
        args:
            image: x-ray image
            fracture_region: (y_start, y_end, x_start, x_end) defining fracture roi
        returns:
            dictionary with fracture-specific metrics
        """
        y1, y2, x1, x2 = fracture_region
        # extract fracture region
        fracture_roi = image[y1:y2, x1:x2]
        # extract surrounding bone region (slightly larger)
        margin = 10
        y1_outer = max(0, y1 - margin)
        y2_outer = min(image.shape[0], y2 + margin)
        x1_outer = max(0, x1 - margin)
        x2_outer = min(image.shape[1], x2 + margin)
        surrounding = image[y1_outer:y2_outer, x1_outer:x2_outer].copy()
        surrounding[y1-y1_outer:y2-y1_outer, x1-x1_outer:x2-x1_outer] = np.nan
        # calculate contrast
        fracture_intensity = np.mean(fracture_roi)
        surrounding_intensity = np.nanmean(surrounding)
        if fracture_intensity + surrounding_intensity != 0:
            contrast = abs(fracture_intensity - surrounding_intensity) / (
                fracture_intensity + surrounding_intensity)
        else:
            contrast = 0.0
        # calculate sharpness in fracture region
        sharpness = ImageMetrics.calculate_variance_of_laplacian(fracture_roi)
        # calculate edge strength
        edge_strength = ImageMetrics.calculate_edge_strength(fracture_roi)
        return {
            'fracture_contrast': contrast,
            'fracture_sharpness': sharpness,
            'fracture_edge_strength': edge_strength,
            'fracture_mean_intensity': fracture_intensity,
            'surrounding_mean_intensity': surrounding_intensity
        }
    @staticmethod
    def get_all_metrics(image: np.ndarray, reference_image: np.ndarray = None) -> Dict[str, float]:
        """
        calculate all available metrics for an image.
        args:
            image: input x-ray image
            reference_image: reference image for comparison metrics (optional)
        returns:
            dictionary containing all metrics
        """
        metrics = {
            'contrast': ImageMetrics.calculate_contrast(image),
            'variance_of_laplacian': ImageMetrics.calculate_variance_of_laplacian(image),
            'gradient_entropy': ImageMetrics.calculate_gradient_entropy(image),
            'edge_strength': ImageMetrics.calculate_edge_strength(image),
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image))
        }
        if reference_image is not None:
            metrics['mse'] = ImageMetrics.calculate_mse(image, reference_image)
            metrics['ssim'] = ImageMetrics.calculate_ssim(image, reference_image)
        return metrics
    @staticmethod
    def get_profile(image: np.ndarray, axis: int = 0, 
                   position: int = None) -> np.ndarray:
        """
        get intensity profile along a line.
        args:
            image: input image
            axis: axis along which to take profile (0=horizontal, 1=vertical)
            position: position along other axis (default: center)
        returns:
            1d intensity profile
        """
        if position is None:
            position = image.shape[1 - axis] // 2
        if axis == 0:
            # horizontal profile
            profile = image[position, :]
        else:
            # vertical profile
            profile = image[:, position]
        return profile
    @staticmethod
    def compare_images(images: list, labels: list, 
                      reference_idx: int = 0) -> Dict[str, list]:
        """
        compare multiple images and return metrics.
        args:
            images: list of images to compare
            labels: list of labels for each image
            reference_idx: index of reference image
        returns:
            dictionary with comparison metrics
        """
        reference = images[reference_idx]
        results = {
            'labels': labels,
            'contrast': [],
            'sharpness': [],
            'edge_strength': [],
            'mse': [],
            'ssim': []
        }
        for i, img in enumerate(images):
            metrics = ImageMetrics.get_all_metrics(img, reference if i != reference_idx else None)
            results['contrast'].append(metrics['contrast'])
            results['sharpness'].append(metrics['variance_of_laplacian'])
            results['edge_strength'].append(metrics['edge_strength'])
            if i != reference_idx:
                results['mse'].append(metrics['mse'])
                results['ssim'].append(metrics['ssim'])
            else:
                results['mse'].append(0.0)
                results['ssim'].append(1.0)
        return results