import imageio.v3 as iio
import numpy as np

# Read all frames as a numpy array
stack = iio.imread("input.tif")  # Shape: (30, H, W)

min_in, max_in = 88, 167
scale = 255.0 / (max_in - min_in)

# Vectorized operation across all frames
stack_output = np.clip((stack.astype(np.float32) - min_in) * scale, 0, 255).astype(
    np.uint8
)

# Write back to TIFF
iio.imwrite("output.tif", stack_output)
