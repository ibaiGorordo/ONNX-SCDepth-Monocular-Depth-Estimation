import cv2
import numpy as np
from imread_from_url import imread_from_url

from scdepth import SCDepth

# Initialize model
model_path='models/sc_depth_v3_nyu_sim.onnx'
depth_estimator = SCDepth(model_path)

# Read inference image
img_url = "https://upload.wikimedia.org/wikipedia/commons/f/f0/Cannery_District_Bozeman_Epic_Fitness_Interior_Wood_Stairs.jpg"
img = imread_from_url(img_url)

# Estimate depth and colorize it
depth_map = depth_estimator(img)
color_depth = depth_estimator.draw_depth()

combined_img = np.hstack((img, color_depth))
cv2.imwrite("doc\img\depth_estimation.png", combined_img)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
cv2.imshow("Estimated depth", combined_img)
cv2.waitKey(0)