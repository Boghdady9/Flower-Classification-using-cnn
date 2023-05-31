import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
import time
from PIL import Image
import matplotlib



# Image Preprocessing
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image_path)

    # Process a PIL image for use in a PyTorch model
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    width, height = pil_image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        pil_image = pil_image.resize((round(aspect_ratio * 256), 256))
    else:
        pil_image = pil_image.resize((256, round(256 / aspect_ratio)))

    # Crop out the center 224x224 portion of the image
    width, height = pil_image.size
    new_width = 224
    new_height = 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    pil_image = pil_image.crop((round(left), round(top), round(right), round(bottom)))

    # Convert color channels to 0-1
    np_image = np.array(pil_image) / 255

    # Normalize the image
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image


# Display the original image (cropped)
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax



# Predict class and probabilities
def predict(np_image, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if gpu else "cpu")

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        images = torch.from_numpy(np_image)
        images = images.unsqueeze(0)
        images = images.type(torch.FloatTensor)
        images = images.to(device) # Move input tensors to the GPU/CPU

        output = model.forward(images)
        ps = torch.exp(output) # get the class probabilities from log-softmax

        probs, indices = torch.topk(ps, topk)
        probs = [float(prob) for prob in probs[0]]
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[int(index)] for index in indices[0]]
        
    return probs, classes



# Get the command line input into the scripts
args = argparse.ArgumentParser(description='Train a neural network')
# Basic usage: python predict.py /path/to/image checkpoint
args.add_argument('image_path', action='store',
                    default = 'flowers/test/1/image_06743.jpg',
                    help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')

args.add_argument('checkpoint', action='store',
                    default = '.',
                    help='Directory of saved checkpoints, e.g., "assets"')

# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
args.add_argument('--top_k', action='store',
                    default = 5,
                    dest='top_k',
                    help='Return top KK most likely classes, e.g., 5')

# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
args.add_argument('--category_names', action='store',
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='File name of the mapping of flower categories to real names, e.g., "cat_to_name.json"')

# Use GPU for inference: python predict.py input checkpoint --gpu
args.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for inference, set a switch to true')

args = args.parse_args()

# print('image_path     = {!r}'.format(parse_results.image_path))
# print('checkpoint     = {!r}'.format(parse_results.checkpoint))
# print('top_k     = {!r}'.format(parse_results.top_k))
# print('category_names     = {!r}'.format(parse_results.category_names))
# print('gpu     = {!r}'.format(parse_results.gpu))

image_path = args.image_path
checkpoint = args.checkpoint
top_k = int(args.top_k)
category_names = args.category_names
gpu = args.gpu

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f, strict=False)

# Load the checkpoint
filepath = checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath, map_location='cpu')
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])

# Image preprocessing
np_image = process_image(image_path)
# imshow(np_image)

# Predict class and probabilities
print(f"Predicting top {top_k} most likely flower names from image {image_path}.")

probs, classes = predict(np_image, model, top_k, gpu)
classes_name = [cat_to_name[class_i] for class_i in classes]

# print("Flower names: ", classes_name)
# print("Probabilities: ", [round(prob, 3) for prob in probs]) 

print("\nFlower name (probability): ")
print("---")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("")
