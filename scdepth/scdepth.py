import time
import cv2
import numpy as np
import onnxruntime


class SCDepth():

    def __init__(self, model_path):
        # Initialize model
        self.model = self.initialize_model(model_path)
        self.min_depth = np.inf
        self.max_depth = -np.inf

    def __call__(self, image):
        return self.estimate_depth(image)

    def initialize_model(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def estimate_depth(self, image):
        input_tensor = self.prepare_input(image)

        outputs = self.inference(input_tensor)

        self.depth_map = self.process_output(outputs)

        return self.depth_map

    def prepare_input(self, img):
        self.img_height, self.img_width = img.shape[:2]

        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (self.input_width, self.input_height),
                               interpolation=cv2.INTER_AREA) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        return img_input[np.newaxis, :, :, :].astype(np.float32)

    def inference(self, input_tensor):
        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        print(time.time() - start)
        return outputs

    def process_output(self, output):
        return np.squeeze(output)

    def draw_depth(self):
        # Normalize estimated depth to color it
        min_depth = self.depth_map.min()
        max_depth = self.depth_map.max()
        self.min_depth = min_depth if min_depth < self.min_depth else self.min_depth
        self.max_depth = max_depth if max_depth > self.max_depth else self.max_depth

        print(self.min_depth, self.max_depth)
        norm_depth_map = 255 * (self.depth_map - min_depth) / (max_depth - min_depth)
        norm_depth_map = 255 - norm_depth_map

        # Normalize and color the image
        color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1),
                                        cv2.COLORMAP_JET)

        # Resize the depth map to match the input image shape
        return cv2.resize(color_depth, (self.img_width, self.img_height))

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.output_shape = model_outputs[0].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    # Initialize model
    model_path = '../models/sc_depth_v3_nyu_sim.onnx'
    depth_estimator = SCDepth(model_path)

    # Read inference image
    img = imread_from_url(
        "https://upload.wikimedia.org/wikipedia/commons/f/f0/Cannery_District_Bozeman_Epic_Fitness_Interior_Wood_Stairs.jpg")

    # Estimate depth and colorize it
    depth_map = depth_estimator(img)
    color_depth = depth_estimator.draw_depth()

    combined_img = np.hstack((img, color_depth))

    cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated depth", combined_img)
    cv2.waitKey(0)
