import argparse
import os
import time
import json

import matplotlib.pyplot as plt 
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict

from PIL import Image

def getArgs():
    p = argparse.ArgumentParser(description='use a neural network to predict type of a plant')
    p.add_argument('input', help='a required argument indicating the input path')
    p.add_argument('checkpoint', help='a required argument indicating the checkpoint path')
    p.add_argument('--arch', help='transfer model to use ')
    p.add_argument('--top_k', help='an argument indicating the top k predictions to display',default=5)
    p.add_argument('--category_names', help='an argument indicating file of the category names',default="cat_to_name.json")
    p.add_argument('--gpu',help='gpu is used',default = False,const =True,nargs='?')
    return p.parse_args()
    

def load_model(path):
    print("loading the model...")
    checkpoint = torch.load(path)
    model = init_transfer_model()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def read_categories():

    return json.loads(open(args.category_names).read())

def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(im)
    coords= im.size
    max_coord = max(coords)
    max_index = coords.index(max_coord)
    min_index = (max_index + 1) % 2 
    aspect_ratio=coords[max_index]/coords[min_index]
    new_coords = [0,0]
    new_coords[min_index] = 256
    new_coords[max_index] = int(256 * aspect_ratio)
    im = im.resize(new_coords)
    x,y = new_coords
    im = im.crop(((x - 244)/2,
                  (y - 244)/2,
                  (x + 244)/2,
                  (y + 244)/2
                 ))
    np_image = np.array(im,dtype='float64')
    np_image = np_image / [255,255,255]
    np_image = ((np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225])
    np_image = np_image.transpose((2, 0, 1))
    return np_image
    

    
def init_transfer_model():
    arch = 'vgg' if args.arch is None else args.arch
    model = models.vgg19(pretrained=True) if arch == 'vgg' else models.densenet121(pretrained=True)
    input_nodes = 25088 if arch == 'vgg' else 1024
    hidden_units = 4096 
    for param in model.parameters():
        param.requires_grad = False
    #REPLACE THE CLASSIFIER
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_nodes, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model

def predict(model , device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_path = args.input
    print(image_path)
    top_k = int(args.top_k)
    model.to(device)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        image.to(device)
        outputs = model.forward(image)
        probs, classes = torch.exp(outputs).topk(top_k)
        return probs[0].tolist(), classes[0].add(1).tolist()

def display_results(probs,classes):
    cat_to_name = read_categories()
    classes2 = [cat_to_name[str(c)]  for c in classes]
    for p , c in zip(probs,classes2):
        print("{:20} prob : {:8.2%}".format(c,p) )
        
def main():
    global args 
    args = getArgs()
    if(not os.path.isfile(args.input)):
        raise Exception('input file does not exist!')
    if(not os.path.isfile(args.checkpoint)):
        raise Exception('checkpoint file does not exist!')
    if(not os.path.isfile(args.category_names)):
        raise Exception('checkpoint file does not exist!')
    if args.gpu  and not torch.cuda.is_available():
        raise Exception("--device gpu option enabled, but no GPU detected")
    device = 'cuda' if args.gpu else  'cpu'
    model = load_model(args.checkpoint)
    probs , classes = predict(model,device)
    display_results(probs,classes)


main()