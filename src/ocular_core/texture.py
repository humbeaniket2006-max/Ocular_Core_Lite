# src/ocular_core/texture.py
import numpy as np
from PIL import Image
from transformers import pipeline

# Load pipeline once at module level to save time
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas")

def extract_normal_map(image_path, save_path, strength=0.5):
    """
    Converts a 2D image into a 3D Normal Map.
    """
    print(f"Processing: {image_path}")
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: {image_path} not found.")
        return

    # Estimate depth
    depth_map = depth_estimator(image)["depth"]
    
    # Calculate gradients
    d_im = np.array(depth_map).astype("float64")
    zy, zx = np.gradient(d_im)
    
    zx = zx * strength
    zy = zy * strength
    
    # Stack and normalize
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    # Convert to RGB
    normal_img = ((normal + 1) / 2 * 255).astype(np.uint8)
    Image.fromarray(normal_img).save(save_path)
    print(f"Normal map saved to: {save_path}")