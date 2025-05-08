import torch
import cv2
import torchvision.transforms as transforms
import argparse

from detection_utils import draw_bboxes

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', required=True, help='path to the input data'
)
args = vars(parser.parse_args())

# computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# init model and utils
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
ssd_model.to(device)
ssd_model.eval()
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# read image
image_path = args['input']
image = cv2.imread(image_path)

# keep the og h & w for resizing of bboxes
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# apply transforms
transformed_image = transform(image)

# convert to torch tensor
tensor = torch.tensor(transformed_image, dtype=torch.float32)

# add batch dim
tensor = tensor.unsqueeze(0).to(device)

# get the detection results
with torch.no_grad():
    detections = ssd_model(tensor)
results_per_input = utils.decode_results(detections)

# threshold >= 0.45
best_results_per_input = [utils.pick_best(results, 0.45) for results in results_per_input]

# get coco dict using utils
classes_to_labels = utils.get_coco_object_dictionary()
image_result = draw_bboxes(image, best_results_per_input, classes_to_labels)
cv2.imshow('Detections', image_result)
cv2.waitKey(0)

# save output image
save_name = args['input'].split('/')[-1]
cv2.imwrite(f"outputs/{save_name}", image_result)