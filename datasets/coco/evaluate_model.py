import torch
import cv2
import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--annotations', required=True, help='path to COCO annotation file'
)
parser.add_argument(
    '--images_dir', required=True, help='directory containing validation images'
)
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("Loading model...")
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
ssd_model.to(device)
ssd_model.eval()
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# Set transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load COCO annotations
coco_gt = COCO(args.annotations)

# Initialize prediction results in COCO format
pred_results = []

# Get image ids from the annotation file
img_ids = coco_gt.getImgIds()
print(f"Found {len(img_ids)} images to evaluate")

# Process each image
for img_id in tqdm(img_ids):
    # Get image info
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join(args.images_dir, img_info['file_name'])
    
    # Read and process image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}")
        continue
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions for later rescaling
    orig_h, orig_w = image.shape[0], image.shape[1]
    
    # Apply transforms
    transformed_image = transform(image)
    
    # Add batch dimension and move to device
    tensor = torch.tensor(transformed_image, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get detections
    with torch.no_grad():
        detections = ssd_model(tensor)
    
    # Decode results
    results_per_input = utils.decode_results(detections)
    best_results = utils.pick_best(results_per_input[0], 0.45)
    
    # Extract bounding boxes, classes, and confidence scores
    bboxes, classes, confidences = best_results
    
    # Convert detections to COCO format
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        
        # Rescale bounding box to original image dimensions
        x1 = int(x1 * orig_w)
        y1 = int(y1 * orig_h)
        x2 = int(x2 * orig_w)
        y2 = int(y2 * orig_h)
        
        # COCO format uses [x, y, width, height]
        width = x2 - x1
        height = y2 - y1
        
        # Create prediction entry
        pred_entry = {
            'image_id': img_id,
            'category_id': int(classes[i]),  # NVIDIA model uses COCO categories
            'bbox': [x1, y1, width, height],
            'score': float(confidences[i])
        }
        pred_results.append(pred_entry)

# Save predictions to file
with open('predictions.json', 'w') as f:
    json.dump(pred_results, f)

print("Predictions saved to predictions.json")

# Evaluate using COCO evaluation
print("Evaluating...")
coco_dt = coco_gt.loadRes('predictions.json')
cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# Extract the main metrics
metrics = {
    'AP (IoU=0.50:0.95)': cocoEval.stats[0],  # mAP
    'AP (IoU=0.50)': cocoEval.stats[1],       # AP50
    'AP (IoU=0.75)': cocoEval.stats[2],       # AP75
    'Precision': np.mean(cocoEval.eval['precision']),
    'Recall': np.mean(cocoEval.eval['recall']) if 'recall' in cocoEval.eval else 'Not available'
}

print("\nEvaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Save metrics to file
with open('evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to evaluation_metrics.json")