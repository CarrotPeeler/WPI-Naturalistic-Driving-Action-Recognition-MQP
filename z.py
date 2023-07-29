from scipy.ndimage import gaussian_filter
import numpy as np

a = np.array([[0.5,0.4,0.1],
              [0.8,0.1,0.1],
              [0.85,0.05,0.1],
              [0.1,0.8,0.1]])

print(gaussian_filter(a, sigma=1))

