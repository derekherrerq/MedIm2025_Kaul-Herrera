"""
virtual x-ray simulation engine
implements beer-lambert law for x-ray attenuation through materials
"""

import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class XRaySimulator:
    """
    simulates x-ray imaging using beer-lambert law.
    i = i0 * exp(-μ * d)
    where i0 is initial intensity, μ is attenuation coefficient, d is distance
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        """
        initialize the x-ray simulator.
        
        args:
            image_size: tuple of (height, width) for output image
        """
        self.image_size = image_size
        self.attenuation_coefficients = self._load_attenuation_data()
        
    def _load_attenuation_data(self) -> Dict[str, Dict[float, float]]:
        """
        load attenuation coefficients (μ) for different materials at various energies.
        values based on nist data (https://physics.nist.gov/physrefdata/xraymasscoef/)
        units: cm²/g, need to multiply by density for linear attenuation
        
        returns:
            dictionary mapping material names to energy-coefficient pairs
        """
        # mass attenuation coefficients at different energies (kev)
        # multiplied by typical densities to get linear attenuation (1/cm)
        return {
            'air': {
                20: 0.0001,
                40: 0.00005,
                60: 0.00003,
                80: 0.00002,
                100: 0.00002,
                150: 0.00001
            },
            'soft_tissue': {
                20: 0.75,
                40: 0.23,
                60: 0.19,
                80: 0.18,
                100: 0.17,
                150: 0.15
            },
            'bone': {
                20: 2.5,
                40: 0.65,
                60: 0.40,
                80: 0.32,
                100: 0.28,
                150: 0.22
            },
            'aluminum': {
                20: 3.2,
                40: 0.85,
                60: 0.50,
                80: 0.38,
                100: 0.32,
                150: 0.25
            }
        }
    
    def get_attenuation_coefficient(self, material: str, energy: float) -> float:
        """
        get attenuation coefficient for a material at given energy.
        interpolates if exact energy not available.
        
        args:
            material: material name
            energy: x-ray energy in kev
            
        returns:
            linear attenuation coefficient in 1/cm
        """
        if material not in self.attenuation_coefficients:
            raise ValueError(f"Material '{material}' not found")
        
        energy_dict = self.attenuation_coefficients[material]
        energies = sorted(energy_dict.keys())
        
        # if exact match
        if energy in energy_dict:
            return energy_dict[energy]
        
        # interpolate
        if energy < energies[0]:
            return energy_dict[energies[0]]
        if energy > energies[-1]:
            return energy_dict[energies[-1]]
        
        # linear interpolation
        for i in range(len(energies) - 1):
            if energies[i] <= energy <= energies[i + 1]:
                e1, e2 = energies[i], energies[i + 1]
                mu1, mu2 = energy_dict[e1], energy_dict[e2]
                return mu1 + (mu2 - mu1) * (energy - e1) / (e2 - e1)
        
        return energy_dict[energies[-1]]
    
    def create_2d_test_phantom(self) -> np.ndarray:
        """
        create a 2d test phantom with various geometric shapes.
        
        returns:
            2d array representing material types (0=air, 1=soft_tissue, 2=bone)
        """
        phantom = np.zeros(self.image_size, dtype=np.uint8)
        h, w = self.image_size
        
        # circle (bone)
        cy, cx = h // 4, w // 4
        y, x = np.ogrid[:h, :w]
        circle_mask = (x - cx)**2 + (y - cy)**2 <= (min(h, w) // 8)**2
        phantom[circle_mask] = 2
        
        # rectangle (soft tissue)
        phantom[h//4:h//2, w//2:3*w//4] = 1
        
        # triangle (bone)
        for i in range(h//2, 3*h//4):
            start = w//4 + (i - h//2) // 2
            end = w//2 - (i - h//2) // 2
            if start < end:
                phantom[i, start:end] = 2
        
        # small circle (high density)
        cy2, cx2 = 3*h//4, 3*w//4
        circle_mask2 = (x - cx2)**2 + (y - cy2)**2 <= (min(h, w) // 12)**2
        phantom[circle_mask2] = 2
        
        return phantom
    
    def create_3d_leg_phantom(self, fracture: bool = False, 
                             fracture_angle: float = 0.0,
                             fracture_width: float = 2.0) -> np.ndarray:
        """
        create a 3d cylindrical leg phantom with optional fracture.
        
        args:
            fracture: whether to include a fracture
            fracture_angle: angle of fracture in degrees
            fracture_width: width of fracture gap in pixels
            
        returns:
            3d array of shape (depth, height, width) with material types
        """
        depth = 200
        phantom = np.zeros((depth, self.image_size[0], self.image_size[1]), dtype=np.uint8)
        
        h, w = self.image_size
        cy, cx = h // 2, w // 2
        
        # outer cylinder (soft tissue) - radius
        outer_radius = min(h, w) // 3
        # inner cylinder (bone) - radius
        inner_radius = outer_radius // 2
        
        y, x = np.ogrid[:h, :w]
        
        for z in range(depth):
            # soft tissue (outer cylinder)
            outer_mask = (x - cx)**2 + (y - cy)**2 <= outer_radius**2
            phantom[z][outer_mask] = 1
            
            # bone (inner cylinder)
            inner_mask = (x - cx)**2 + (y - cy)**2 <= inner_radius**2
            phantom[z][inner_mask] = 2
            
            # add fracture if requested
            if fracture:
                fracture_z_start = depth // 3
                fracture_z_end = 2 * depth // 3
                
                if fracture_z_start <= z <= fracture_z_end:
                    # create fracture line
                    angle_rad = np.radians(fracture_angle)
                    
                    # rotate coordinates
                    xr = (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad)
                    
                    # create gap in bone
                    fracture_mask = (np.abs(xr) < fracture_width / 2) & inner_mask
                    phantom[z][fracture_mask] = 1  # replace bone with soft tissue in gap
        
        return phantom
    
    def apply_beer_lambert_law(self, phantom: np.ndarray, energy: float,
                               projection_axis: int = 0) -> np.ndarray:
        """
        apply beer-lambert law to simulate x-ray attenuation.
        
        args:
            phantom: 2d or 3d array of material types
            energy: x-ray energy in kev
            projection_axis: axis along which to project (for 3d phantoms)
            
        returns:
            2d intensity map after attenuation
        """
        material_map = {
            0: 'air',
            1: 'soft_tissue',
            2: 'bone',
            3: 'aluminum'
        }
        
        if phantom.ndim == 2:
            # 2d phantom - direct calculation
            thickness = np.ones_like(phantom, dtype=float) * 10.0  # assume 10cm thickness
            attenuation = np.zeros_like(phantom, dtype=float)
            
            for mat_id, mat_name in material_map.items():
                mask = phantom == mat_id
                mu = self.get_attenuation_coefficient(mat_name, energy)
                attenuation[mask] = mu * thickness[mask]
            
        else:
            # 3d phantom - integrate along projection axis
            attenuation = np.zeros(phantom.shape[1:], dtype=float)
            pixel_thickness = 0.1  # cm per pixel along projection axis
            
            for mat_id, mat_name in material_map.items():
                mu = self.get_attenuation_coefficient(mat_name, energy)
                mat_mask = phantom == mat_id
                thickness_map = np.sum(mat_mask, axis=projection_axis) * pixel_thickness
                attenuation += mu * thickness_map
        
        # apply beer-lambert law: i = i0 * exp(-μd)
        initial_intensity = 1.0
        intensity = initial_intensity * np.exp(-attenuation)
        
        return intensity
    
    def simulate_geometry(self, image: np.ndarray, 
                         source_distance: float = 100.0,
                         object_distance: float = 50.0,
                         film_distance: float = 150.0) -> np.ndarray:
        """
        simulate geometric magnification and blur effects.
        
        args:
            image: input intensity image
            source_distance: distance from source to reference plane (cm)
            object_distance: distance from source to object (cm)
            film_distance: distance from source to film (cm)
            
        returns:
            image with geometric effects applied
        """
        # magnification factor: m = (source-to-film) / (source-to-object)
        magnification = film_distance / object_distance
        
        # resize image based on magnification
        from scipy.ndimage import zoom
        output_image = zoom(image, magnification, order=1)
        
        # crop or pad to original size
        h, w = self.image_size
        oh, ow = output_image.shape
        
        if oh > h or ow > w:
            # crop
            start_h = (oh - h) // 2
            start_w = (ow - w) // 2
            output_image = output_image[start_h:start_h+h, start_w:start_w+w]
        else:
            # pad
            pad_h = (h - oh) // 2
            pad_w = (w - ow) // 2
            padded = np.zeros((h, w))
            padded[pad_h:pad_h+oh, pad_w:pad_w+ow] = output_image
            output_image = padded
        
        # apply slight blur based on geometry (penumbra effect)
        from scipy.ndimage import gaussian_filter
        blur_amount = max(0.1, (film_distance - object_distance) / source_distance)
        output_image = gaussian_filter(output_image, sigma=blur_amount)
        
        return output_image
    
    def apply_beam_angle(self, phantom: np.ndarray, angle: float) -> np.ndarray:
        """
        simulate x-ray beam at an angle.
        
        args:
            phantom: input phantom (2d or 3d)
            angle: beam angle in degrees (0 = perpendicular)
            
        returns:
            rotated phantom
        """
        from scipy.ndimage import rotate
        
        if phantom.ndim == 2:
            return rotate(phantom, angle, reshape=False, order=1)
        else:
            # rotate each slice
            rotated = np.zeros_like(phantom)
            for i in range(phantom.shape[0]):
                rotated[i] = rotate(phantom[i], angle, reshape=False, order=1)
            return rotated
    
    def add_noise(self, image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        add poisson noise to simulate quantum noise in x-ray imaging.
        
        args:
            image: input image
            noise_level: noise standard deviation
            
        returns:
            noisy image
        """
        # scale image for poisson noise
        scaled = image * 1000
        noisy = np.random.poisson(scaled) / 1000.0
        
        # add gaussian noise
        gaussian_noise = np.random.normal(0, noise_level, image.shape)
        noisy = noisy + gaussian_noise
        
        return np.clip(noisy, 0, 1)
    
    def simulate_xray(self, phantom_type: str = '2d_test',
                      energy: float = 60.0,
                      source_distance: float = 100.0,
                      object_distance: float = 50.0,
                      film_distance: float = 150.0,
                      beam_angle: float = 0.0,
                      fracture: bool = False,
                      fracture_angle: float = 0.0,
                      fracture_width: float = 2.0,
                      noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        complete x-ray simulation pipeline.
        
        args:
            phantom_type: type of phantom ('2d_test' or '3d_leg')
            energy: x-ray energy in kev
            source_distance: distance from source to reference (cm)
            object_distance: distance from source to object (cm)
            film_distance: distance from source to film (cm)
            beam_angle: beam angle in degrees
            fracture: include fracture (3d leg only)
            fracture_angle: fracture angle in degrees
            fracture_width: fracture width in pixels
            noise_level: noise level
            
        returns:
            tuple of (phantom, xray_image)
        """
        # create phantom
        if phantom_type == '2d_test':
            phantom = self.create_2d_test_phantom()
            projection_axis = None
        else:
            phantom = self.create_3d_leg_phantom(fracture, fracture_angle, fracture_width)
            projection_axis = 0
        
        # apply beam angle
        if beam_angle != 0:
            phantom = self.apply_beam_angle(phantom, beam_angle)
        
        # apply beer-lambert law
        if projection_axis is not None:
            intensity = self.apply_beer_lambert_law(phantom, energy, projection_axis)
        else:
            intensity = self.apply_beer_lambert_law(phantom, energy)
        
        # apply geometric effects
        intensity = self.simulate_geometry(intensity, source_distance, 
                                          object_distance, film_distance)
        
        # add noise
        intensity = self.add_noise(intensity, noise_level)
        
        # for display purposes, return 2d slice of phantom if 3d
        if phantom.ndim == 3:
            phantom_display = phantom[phantom.shape[0] // 2]
        else:
            phantom_display = phantom
        
        return phantom_display, intensity