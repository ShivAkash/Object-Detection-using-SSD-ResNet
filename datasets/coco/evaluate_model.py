import torch
import cv2
import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as transforms
import torchvision.ops as ops
from tqdm import tqdm
import argparse
import sys

# Add parent directory to path for importing custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--annotations', required=True, help='path to COCO annotation file'
)
parser.add_argument(
    '--images_dir', required=True, help='directory containing validation images'
)
parser.add_argument(
    '--threshold', type=float, default=0.05, help='confidence threshold for detections - lower for higher recall'
)
parser.add_argument(
    '--iou_threshold', type=float, default=0.5, help='IoU threshold for NMS'
)
parser.add_argument(
    '--use_tta', action='store_true', help='Use test-time augmentation'
)
parser.add_argument(
    '--multi_scale', action='store_true', help='Use multi-scale detection'
)
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print("Loading model...")
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
ssd_model.to(device)
ssd_model.eval()
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# Set transforms - ensure these match exactly what the model was trained with
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Define test-time augmentation transforms
tta_transforms = [
    # Original image
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

# Load COCO annotations
coco_gt = COCO(args.annotations)

# Initialize prediction results in COCO format
pred_results = []

# Get image ids from the annotation file
img_ids = coco_gt.getImgIds()
print(f"Found {len(img_ids)} images to evaluate")

# Get COCO categories - used for class mapping
categories = coco_gt.loadCats(coco_gt.getCatIds())
coco_category_ids = [cat['id'] for cat in categories]
print(f"COCO has {len(coco_category_ids)} categories")

# Load the correct class mapping between NVIDIA SSD and COCO
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
category_path = os.path.join(root_dir, 'category_names.txt')

# Use direct mapping between COCO categories and model outputs
coco_labels = []
with open(category_path, 'r') as f:
    for line in f:
        coco_labels.append(line.strip())

# Create direct model ID to COCO category ID mapping
# The NVIDIA SSD model uses category indices that align with COCO
model_to_coco_mapping = {}
for i, cat in enumerate(categories):
    model_class_id = i + 1  # Model indices start at 1
    model_to_coco_mapping[model_class_id] = cat['id']

print(f"Created mapping for {len(model_to_coco_mapping)} classes out of {len(categories)} COCO categories")

# Apply traditional NMS with class separation
def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    # Apply NMS per class to avoid removing objects of different classes
    unique_classes = torch.unique(classes)
    keep_boxes = []
    keep_scores = []
    keep_classes = []
    
    for cls in unique_classes:
        class_mask = (classes == cls)
        cls_boxes = boxes[class_mask]
        cls_scores = scores[class_mask]
        
        if len(cls_boxes) > 0:
            # Apply NMS
            keep_indices = ops.nms(cls_boxes, cls_scores, iou_threshold)
            
            # Add kept boxes to results
            keep_boxes.append(cls_boxes[keep_indices])
            keep_scores.append(cls_scores[keep_indices])
            keep_classes.append(torch.full((len(keep_indices),), cls))
    
    # Combine results from all classes
    if keep_boxes:
        return torch.cat(keep_boxes), torch.cat(keep_scores), torch.cat(keep_classes)
    else:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

# Helper function for test-time augmentation
def apply_tta(image, model, utils):
    """Apply test-time augmentation and combine predictions"""
    all_bboxes = []
    all_classes = []
    all_confidences = []
    
    for transform in tta_transforms:
        # Apply current transform
        transformed = transform(image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            detections = model(transformed)
        
        # Decode results
        results = utils.decode_results(detections)[0]
        bboxes, classes, confidences = results
        
        # For horizontal flip, adjust bounding box coordinates
        if transform == tta_transforms[1]:  # Horizontal flip transform
            for i in range(len(bboxes)):
                # Flip x-coordinates: x' = 1 - x for normalized coordinates
                x1, y1, x2, y2 = bboxes[i]
                bboxes[i] = [1.0 - x2, y1, 1.0 - x1, y2]
        
        # Convert to numpy arrays to avoid list issues
        all_bboxes.extend(np.array(bboxes).tolist())
        all_classes.extend(np.array(classes).tolist())
        all_confidences.extend(np.array(confidences).tolist())
    
    return all_bboxes, all_classes, all_confidences

# Process each image in batches
batch_size = 8
for i in tqdm(range(0, len(img_ids), batch_size)):
    batch_ids = img_ids[i:i+batch_size]
    valid_images = []
    
    # Process each image
    for img_id in batch_ids:
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
        
        # Multi-scale detection if enabled
        all_bboxes = []
        all_classes = []
        all_confidences = []
        
        # Perform regular detection or TTA if enabled
        if args.use_tta:
            bboxes, classes, confidences = apply_tta(image, ssd_model, utils)
            all_bboxes.extend(bboxes)
            all_classes.extend(classes)
            all_confidences.extend(confidences)
        else:
            # Apply standard transform
            transformed_image = transform(image).unsqueeze(0).to(device)
            
            # Get detections
            with torch.no_grad():
                detections = ssd_model(transformed_image)
            
            # Decode results for this image
            results = utils.decode_results(detections)[0]
            bboxes, classes, confidences = results
            all_bboxes.extend(bboxes)
            all_classes.extend(classes)
            all_confidences.extend(confidences)
        
        # Add multi-scale detections if enabled
        if args.multi_scale:
            # Define multiple scales for detection
            scales = [0.8, 1.2]  # 80% and 120% of original size
            
            for scale in scales:
                # Resize image with scale factor
                h, w = int(orig_h * scale), int(orig_w * scale)
                resized = cv2.resize(image, (w, h))
                
                # Apply standard transform
                transformed = transform(resized).unsqueeze(0).to(device)
                
                # Get detections
                with torch.no_grad():
                    detections = ssd_model(transformed)
                
                # Decode results
                results = utils.decode_results(detections)[0]
                ms_bboxes, ms_classes, ms_confidences = results
                
                # Adjust bounding box coordinates for the scaled image
                for i in range(len(ms_bboxes)):
                    x1, y1, x2, y2 = ms_bboxes[i]
                    # Normalize to original image size
                    ms_bboxes[i] = [x1, y1, x2, y2]
                
                all_bboxes.extend(ms_bboxes)
                all_classes.extend(ms_classes)
                all_confidences.extend(ms_confidences)
        
        # Convert to torch tensors for NMS
        if not all_bboxes:
            continue
            
        bboxes_tensor = torch.tensor(all_bboxes, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.int64)
        confidences_tensor = torch.tensor(all_confidences, dtype=torch.float32)
        
        # Filter by confidence threshold
        mask = confidences_tensor >= args.threshold
        if torch.sum(mask) == 0:
            continue
            
        bboxes_tensor = bboxes_tensor[mask]
        classes_tensor = classes_tensor[mask]
        confidences_tensor = confidences_tensor[mask]
        
        # Apply our custom NMS implementation
        nms_bboxes, nms_scores, nms_classes = apply_nms(
            bboxes_tensor, confidences_tensor, classes_tensor, args.iou_threshold
        )
        
        # Skip if no detections after NMS
        if len(nms_bboxes) == 0:
            continue
            
        # Convert detections to COCO format
        for i in range(len(nms_bboxes)):
            # Get normalized coordinates (0-1)
            x1, y1, x2, y2 = nms_bboxes[i].tolist()
            
            # Convert to absolute pixel coordinates
            x1_abs = x1 * orig_w
            y1_abs = y1 * orig_h
            x2_abs = x2 * orig_w
            y2_abs = y2 * orig_h
            
            # COCO format uses [x, y, width, height]
            width = x2_abs - x1_abs
            height = y2_abs - y1_abs
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
                
            # Get class and map to COCO category ID
            model_class_id = int(nms_classes[i].item())
            
            # Map model class ID to COCO category ID using our improved mapping
            if model_class_id in model_to_coco_mapping:
                category_id = model_to_coco_mapping[model_class_id]
            else:
                # Skip classes not in the mapping
                continue
            
            # Create prediction entry
            pred_entry = {
                'image_id': img_id,
                'category_id': category_id,
                'bbox': [float(x1_abs), float(y1_abs), float(width), float(height)],
                'score': float(nms_scores[i].item())
            }
            pred_results.append(pred_entry)

print(f"Generated {len(pred_results)} predictions across {len(img_ids)} images")

# Save predictions to file
predictions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.json')
with open(predictions_file, 'w') as f:
    json.dump(pred_results, f)

print(f"Predictions saved to {predictions_file}")

# Evaluate using COCO evaluation
print("Evaluating...")
if len(pred_results) == 0:
    print("ERROR: No predictions were generated. Please check the threshold and image paths.")
    exit(1)
    
coco_dt = coco_gt.loadRes(predictions_file)
cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')

# Configure evaluation parameters
cocoEval.params.imgIds = img_ids  # Restrict to images we processed
cocoEval.params.maxDets = [1, 10, 100]  # Standard COCO detection counts
cocoEval.params.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
cocoEval.params.areaRngLbl = ['all', 'small', 'medium', 'large']

# Run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# Extract the main metrics
metrics = {
    'AP (IoU=0.50:0.95)': float(cocoEval.stats[0]),  # mAP
    'AP (IoU=0.50)': float(cocoEval.stats[1]),       # AP50
    'AP (IoU=0.75)': float(cocoEval.stats[2]),       # AP75
    'AP (small)': float(cocoEval.stats[3]),          # AP for small objects 
    'AP (medium)': float(cocoEval.stats[4]),         # AP for medium objects
    'AP (large)': float(cocoEval.stats[5]),          # AP for large objects
    'AR (max=1)': float(cocoEval.stats[6]),          # AR with 1 detection per image
    'AR (max=10)': float(cocoEval.stats[7]),         # AR with 10 detections per image
    'AR (max=100)': float(cocoEval.stats[8]),        # AR with 100 detections per image
    'AR (small)': float(cocoEval.stats[9]),          # AR for small objects
    'AR (medium)': float(cocoEval.stats[10]),        # AR for medium objects
    'AR (large)': float(cocoEval.stats[11])          # AR for large objects
}

print("\nEvaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Save metrics to file
metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_file}")