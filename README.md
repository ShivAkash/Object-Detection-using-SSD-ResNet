# Object Detection using NVIDIA SSD ResNet

This project uses NVIDIA's pre-trained SSD (Single Shot MultiBox Detector) with a ResNet backbone to detect objects in images. 


## Installation

1. Clone the repository.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Install CUDA-supported PyTorch (recommended):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
For a command customized to your machine, visit: https://pytorch.org/get-started/locally/

## Requirements

- Python 3.6 or higher
- PyTorch
- Torchvision
- OpenCV
- scikit-image

## Usage

Run the detection script on an image with:

```bash
python ssd_resnet_image.py -i input/image_0.jpg
```

You can view the resulting image in the output directory






