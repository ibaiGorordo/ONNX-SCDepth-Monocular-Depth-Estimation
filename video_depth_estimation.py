import cv2
from cap_from_youtube import cap_from_youtube, list_video_streams
import numpy as np

from scdepth import SCDepth

# Initialize video
# cap = cv2.VideoCapture("video.mp4")


youtube_url = 'https://youtu.be/e0IjlkU-pX0'
cap = cap_from_youtube(youtube_url, '1080p')
start_time = 10 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize model
model_path='models/sc_depth_v3_nyu_sim.onnx'
depth_estimator = SCDepth(model_path)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	# Estimate depth and colorize it
	depth_map = depth_estimator(frame)
	color_depth = depth_estimator.draw_depth()

	combined_img = cv2.addWeighted(frame, 0.5, color_depth, 0.5, 0)

	cv2.imshow("Estimated depth", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
