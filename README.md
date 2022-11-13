# ONNX-PackNet-SfM
Python scripts for performing monocular depth estimation using the SC_Depth model in ONNX

![SC_Depth monocular depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-SCDepth-Monocular-Depth-Estimation/blob/main/doc/img/out.png)
*Original image:https://commons.wikimedia.org/wiki/File:Cannery_District_Bozeman_Epic_Fitness_Interior_Wood_Stairs.jpg*

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-SCDepth-Monocular-Depth-Estimation.git
cd ONNX-YOLOv7-Object-Detection
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`


# ONNX model
I don't provide the model, but you can easily convert the Pytorch model by placing the following code in the [line #80 of inference.py in the original repository](https://github.com/JiawangBian/sc_depth_pl/blob/main/inference.py#L80):
```
model_name = "sc_depth_v3_nyu.onnx"
torch.onnx.export(model,  # model being run
                  tensor_img,  # model input (or a tuple for multiple inputs)
                  model_name,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=16)
```     

Then you can simplify the ONNX model using [ONNX-Simplifier](https://github.com/daquexian/onnx-simplifier):
```
onnxsim sc_depth_v3_nyu.onnx sc_depth_v3_nyu_sim.onnx
```

Finally, copy the simplified ONNX model file to the [**models** folder](https://github.com/ibaiGorordo/ONNX-SCDepth-Monocular-Depth-Estimation/blob/main/models).

# Original Pytorch model
The Pytorch pretrained models were taken from the [original repository](https://github.com/JiawangBian/sc_depth_pl).
 
# Examples

 * **Image inference**:
 
 ```
 python image_depth_estimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python video_depth_estimation.py
 ```
 
 * **Webcam inference**:
 
 ```
 python webcam_depth_estimation.py
 ```
 
# [Inference video Example](https://youtu.be/yjjADhCTITk) 
 ![SC_Depth monocular depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-SCDepth-Monocular-Depth-Estimation/blob/main/doc/img/sc_depth_video.gif)

*Original video: https://youtu.be/e0IjlkU-pX0*

# References:
* SC_Depth model: https://github.com/JiawangBian/sc_depth_pl
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* Original papers: https://arxiv.org/abs/2211.03660
