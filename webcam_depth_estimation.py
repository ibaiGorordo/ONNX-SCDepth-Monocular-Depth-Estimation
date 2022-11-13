import cv2
import numpy as np

from scdepth import SCDepth

# Initialize camera
cap = cv2.VideoCapture(1)

# Initialize model
model_path='models/sc_depth_v3_nyu_sim.onnx'
depth_estimator = SCDepth(model_path)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the video
	ret, frame = cap.read()
	if not ret:	
		break
	
	# Estimate depth and colorize it
	depth_map = depth_estimator(frame)
	color_depth = depth_estimator.draw_depth()

	combined_img = np.hstack((frame, color_depth))
	
	cv2.imshow("Estimated depth", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

