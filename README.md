# Image Classification Inference Comparison

## Overview

This project demonstrates the inference of an image classification model using different frameworks and APIs. The comparison includes execution times for a 200x200 grayscale image on an Intel Celeron CPU, averaged over 20 runs.

### Inference Times

| Framework / API             | Average Time (seconds) |
|----------------------------|------------------------|
| `make_inference`           | 0.053                  |
| OpenCV Python API          | 0.126                  |
| ONNX Runtime Python API    | 0.030                  |
| OpenVINO                   | 0.032                  |

All inferences were performed on **CPU**.

---

## Model Description

- **Architecture**: ResNet-50
- **Input Size**: `(1, 1, 200, 200)` — batch size 1, single-channel (grayscale), image resolution 200x200
- **Task**: Image classification
- **Dataset**: Custom dataset or similar to ImageNet subset
- **Model Format**: ONNX (`*.onnx`)
- **Pretrained Weights**: Loaded from PyTorch's `models.resnet50(pretrained=True)`

---

## Getting Started

### Requirements

- Python >= 3.7
- PyTorch
- ONNX
- OpenCV (`cv2`)
- ONNX Runtime
- OpenVINO Toolkit (2022.3 or newer)
- Netron (for visualizing ONNX models)

Install dependencies:
```bash
pip install torch onnx opencv-python onnxruntime openvino-dev
```

---

## Step-by-step Instructions

### 1. Load Model and Create Input

```python
import torch

def create_input():
    input_shape = (1, 1, 200, 200)
    return torch.randn(input_shape)
```

### 2. Export Model to ONNX

```python
def export_onnx(model):
    model.eval()
    dummy_input = create_input()
    model_path = "source_model_name.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    print(f"Model exported to {model_path}")
```

> 📌 You can verify the correctness of the ONNX export using [Netron](https://netron.app/) by opening the exported `.onnx` file.

---

## Inference Pipelines

### A. OpenCV Python API

#### Steps:

1. Load model:
   ```python
   import cv2

   onnx_model_path = "source_model_name.onnx"
   opencv_net = cv2.dnn.readNetFromONNX(onnx_model_path)
   ```

2. Prepare input:
   ```python
   input_img = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)
   input_blob = cv2.dnn.blobFromImage(image=input_img, size=(200, 200))
   ```

3. Run inference:
   ```python
   opencv_net.setInput(input_blob)
   out = opencv_net.forward()
   ```

---

### B. ONNX Runtime Python API

#### Steps:

1. Load model:
   ```python
   import onnxruntime as ort

   session = ort.InferenceSession('source_model_name.onnx', providers=['CPUExecutionProvider'])
   ```

2. Get input/output info:
   ```python
   input_nodes = session.get_inputs()
   input_names = [node.name for node in input_nodes]
   input_types = [node.type for node in input_nodes]

   output_nodes = session.get_outputs()
   output_names = [node.name for node in output_nodes]
   ```

3. Run inference:
   ```python
   input_data = create_input().numpy()  # Convert torch tensor to numpy array if needed
   output_tensors = session.run([], input_feed={input_names[0]: input_data})
   ```

---

### C. OpenVINO Inference

#### Steps:

1. Initialize OpenVINO Core:
   ```python
   from openvino.runtime import Core

   ie = Core()
   ```

2. Load ONNX model:
   ```python
   onnx_model_path = "source_model_name.onnx"
   model_onnx = ie.read_model(model=onnx_model_path)
   compiled_model = ie.compile_model(model=model_onnx, device_name="CPU")
   ```

3. Get input/output layers:
   ```python
   input_layer = compiled_model.input(0)
   output_layer = compiled_model.output(0)
   ```

4. Run inference:
   ```python
   input_data = create_input().numpy()  # Convert torch tensor to numpy array
   request = compiled_model.create_infer_request()
   request.infer(inputs={input_layer.any_name: input_data})
   result = request.get_output_tensor(output_layer.index).data
   ```

---

## Notes

- All models run on **CPU**.
- Input shape used across all frameworks: `(1, 1, 200, 200)`
- Preprocessing steps should match training pipeline (normalization, resizing, etc.)
- ONNX export uses `opset_version=11`
- For debugging purposes, use [Netron](https://netron.app/) to inspect the ONNX graph structure.

---

## Summary

- **Fastest Inference**: ONNX Runtime (0.030 s)
- **Best Trade-off between Accuracy and Speed**: OpenVINO (0.032 s)
- **Slowest Inference**: OpenCV DNN (0.126 s)

All methods are valid for deploying a PyTorch model via ONNX to production-ready APIs with minimal performance overhead.

---

## References

- [Exporting a Model from PyTorch to ONNX](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [OpenCV DNN Module - ONNX Inference](https://docs.opencv.org/4.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html)
- [OpenCV C++ DNN Module - ONNX Inference](https://docs.opencv.org/4.x/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html)
- [ONNX Runtime Python API Documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/get-started/with-cpp.html)
- [OpenVINO Python API Guide](https://docs.openvino.ai/2022.3/notebooks/002-openvino-api-with-output.html)
- [OpenVINO Demos](https://docs.openvino.ai/2022.3/openvino_docs_get_started_get_started_demos.html)

--- 

## To Do List

- [x] Specify target device: CPU  
- [x] Describe model architecture  
- [x] Load pretrained weights  
- [x] Generate random input example  
- [x] Export model to ONNX  
- [x] Implement inference with OpenCV  
- [x] Implement inference with ONNX Runtime  
- [x] Implement inference with OpenVINO  

--- 

## License

This project is provided under the MIT license. See `LICENSE.md` for more details.
