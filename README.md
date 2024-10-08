
# YOLOv11 Object Detection and TensorRT Export

## Overview

This project demonstrates how to use YOLOv8 for object detection and export the model to TensorRT for efficient inference. The model is trained on a dataset and can be exported to TensorRT for deployment. This README includes instructions to set up the environment, run inference, and export the model.

## Requirements

You need the following dependencies to run this project:

```bash
pip install ultralytics
```

Other required libraries:

- torch
- torchvision
- numpy
- matplotlib
- opencv-python
- PIL (Python Imaging Library)

### Additional Dependencies

- `ultralytics-thop`: For model export functionalities.

To install:

```bash
pip install ultralytics ultralytics-thop
```

## Usage

### Model Training and Export

This example shows how to load a YOLOv8 model, run inference, and export it to TensorRT format:

```python
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/path/to/your/weights/best.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'
```

### Handling Errors during Export

If you encounter an error during export (e.g., running on CPU instead of GPU), you can resolve it by setting the device:

```python
# Ensure export is done on a GPU
model.export(format="engine", device="cuda")
```

### Inference Example

Here is an example of running inference using the YOLOv8 model:

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load a model
model = YOLO("/path/to/weights/best.pt")

# Run inference on an image
results = model(["/path/to/image.jpg"])

# Display the results with bounding boxes
for result in results:
    boxes = result.boxes
    # Process boxes and display results
    # Your code for displaying or saving images with detected boxes
```

## Results

Results of the inference can be displayed or saved as images with bounding boxes indicating the detected objects.

## License

This project is licensed under the MIT License.
