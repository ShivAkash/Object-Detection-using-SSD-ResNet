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

## Example Outputs

- image_0.jpg:
  
![image_0](https://github.com/user-attachments/assets/6690aada-f405-4f38-8d9a-11221e6b24a5)
> *get it? hehe!*


- image_1.jpg:
  
![image_1](https://github.com/user-attachments/assets/3ce97290-c3f9-4f4f-b6d3-c93d3b097d9c)
> *GREAT SCOTT!*


- image_2.jpg:
  
![image_2](https://github.com/user-attachments/assets/6d4fbc89-705a-43df-adba-d2a398cacb72)
> *well technically its a cat ☝️🤓*


## View Weights

To view weights and other params of the model, run:
```bash
python view_weights.py ssd_resnet_weights.pth
```

## Evaluation metrics

To evaluate mAP, precision, recall:
```bash
python evaluate_model.py --annotations path/to/annotations.json --images_dir path/to/validation/images
```
you can get the files from the kaggle dataset: https://www.kaggle.com/datasets/sabahesaraki/2017-2017
from there, download **annotations_trainval2017** and **val2017**



