
# YOLOv1 FACE DETECTION

This project implements a custom yolov1 architecture and detects faces from cam feed. It covers complete development pipeline from data seperation, model training, testing, and deployment in pytorch, libtorch and opencv




## Features

- A custom YoloV1 implementation in PyTorch
- Single Class face detection with easy development pipeline to extend to multiclass detection
- Data agumentation using Albumentations
- Bounding Box regression and Confidence prediction
- Non-Maximum Suppression (NMS) for duplicate removal
- Model evaluation using Precision, Recall and F1 Score
- A simple C++ inference in libtorch and opencv built using CMake


## Installation

- For model devleopment

```bash
  git clone https://github.com/PranavTripathi1021/YoloV1-Face-Detection
  cd Face-Detection
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
```

- For deployment
```bash
  sudo apt update
  sudo apt install build-essential cmake
  sudo apt install libopencv-dev
```
      Install libtorch following - https://pytorch.org/get-started/locally
      To install cuda toolkit - https://developer.nvidia.com/cuda-downloads
      To install cuDNN - https://developer.nvidia.com/cudnn-downloads

      Make sure your libtorch version supports the cuda and cuDNN versions
      Configure the cmake presets according to your cuda paths





    
## Model Architecture

This detector follows YoloV1 grid-based prediction approch

This Architecture only allows RGB images of size 448x448

Input image is divied into 7x7 grid

Each grid predicts:\
    - 1 Class probability (Face)\
    - 2 Box confidence scores\
    - 2 bounding boxes, each contains 4 coordinates (all in order)\

Final output shape is 7x7x11

Bounding-box predictions are decoded and filtered using confidence thresholding and Non-Maximum Suppression.


## Dataset and Training

Datasets used for training and validation are
    - https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset
    - https://www.kaggle.com/datasets/nguyenquocdungk16hl/bg-20o 
    I used the second datasets 1/testval to reduce false positves. Move the downloaded images to training folder

### Training Pipeline

- Image Resizing and normalization
- Data augmentation using Albumentations
- Validation and checkpoint saving

### Data augmentation

- Horizontal flipping
- Random affine transformations
- Color jittering
- Image normalization
### Loss evaluation

![Loss Image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-4ZPvW9q9sxl9vWVCBzo2nUUzMGrgGZ37oiBJrjAo28veUUvm8i9dxeA&s=10)
Image taken from pyimagesearch.com

### Optimizer
AdamW
- Learning rates
    
    - Epoch 1-20  -> 5e-5
    - Epoch 20-30  ->  1e-5
    - Epoch 30-90  ->  5e-6
- Weight Decay -> 5e-4
- Beta -> (0.9, 0.999)

### Scheduler
ReduceLROnPlateau
- Patience -> 4
- Factor -> 0.5

### Training and Validation Loss
![Training and Validataion Loss](https://i.ibb.co/MxjLpng2/train-val-loss.png)
`All losses per epoch are not available `
## Evaluation

Dataset used for evaluation is - https://www.kaggle.com/datasets/freak2209/face-data

This model is evalated using
| Attribute | Details |
| :--- | :--- |
|Precision|Proportion of predicted faces that are correct| 
|Recall|Proportion of actual faces successfully detected|
|F1 Score|Harmonic mean of precision and recall|


| Precision | Recall | F1 Score |
| :---      | :---:  | ---:     |
| 0.9982      | 0.8006  | 0.8885     |


## Inference

The inference pipeline consists of

- Image resizing
- Model prediction
- Bounding Box detection
- Confidence filtering
- Non maximum suppression
- Bounding box evaluation
## Deployment

A C++ application is developed using

- Libtorch -  PyTorch's C++ frontend for model inference
- OpenCV - Image processing and bounding-box visualization
- CMake - Build system and dependency management

## Project Structure

The directories of should be made as such or the paths in code should be modified

```text
├── app/
│   ├── model/                # Core C++ application models / logic
│   ├── thirdparty/           # Third-party libraries and dependencies
│   ├── CMakeLists.txt        # Build configuration for CMake
│   ├── CMakePresets.json     # Configuration presets for CMake builds
│   └── main.cpp              # C++ application entry point
├── assets/                   # Training Loss Image
├── code/                     # Python source files for YOLOv1 training & utility
│   ├── data_setup.py         # Script to prepare and preprocess data
│   ├── export.py             # Model deployment and weight export utility
│   ├── lossYoloV1.py         # Custom YOLOv1 loss function implementation
│   ├── model_builder.py      # Architecture definitions for the model
│   ├── model_testing.py      # Script to run validation/testing routines
│   └── train.py              # Main model training loop execution
├── data/                     # Dataset storage for training and testing
├── model/                    # Saved models and checkpoints
├── notebook/                 # Jupyter notebooks for experimentation & visualization
│   ├── code_visual.ipynb     # Notebook for visualizing code execution/outputs
│   ├── data_setup.ipynb      # Prototyping data pipeline and transforms
│   ├── data_visual.ipynb     # Dataset exploratory analysis and visualization
│   ├── loss.ipynb            # Notebook for verifying and testing the loss function
│   ├── model_builder.ipynb   # Step-by-step model building and testing
│   └── model_evaluation.ipynb# Model performance analysis and metrics
├── requirements.txt          # Python packages and project dependencies
└── test.py                   # Main Python entry point for quick testing/inference
```

## Limitations

- Single Class detection - The model is trained to recognise only faces.
- Grid-cell limitations - The 7×7 grid-based target representation discards multiple faces assigned to the same grid cell during target generation.
- Threshold sensitivity - Detection results depend on the confidence threshold and NMS threshold, which influence the precision and recall trade-off.
- Small faces: Detecting very small or distant faces may be challenging due to the model's input resolution and grid-based architecture.


## Author

[Pranav Tripathi](https://github.com/PranavTripathi1021)\
Developed as a personal machine learning and computer vision project.

