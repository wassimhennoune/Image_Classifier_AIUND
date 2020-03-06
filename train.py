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


def getArgs():
    p = argparse.ArgumentParser(description='Train a neural network for plants prediction')
    p.add_argument('data_directory', help='a required argument indicating the data directory')
    p.add_argument('--save_dir', help='directory to save the final trained neural network.')
    p.add_argument('--arch', help='transfer model to use ')
    p.add_argument('--learning_rate', help='value of the learning rate')
    p.add_argument('--hidden_units', help='number of hidden units')
    p.add_argument('--epochs', help='number of epochs',default=3)
    p.add_argument('--gpu',help='gpu is used',default = False,const =True,nargs='?')
    return p.parse_args()
  

def init_transfer_model():
    arch = 'vgg' if args.arch is None else args.arch
    model = models.vgg19(pretrained=True) if arch == 'vgg' else models.densenet121(pretrained=True)
    input_nodes = 25088 if arch == 'vgg' else 1024
    hidden_units = 4096 if args.hidden_units is None else args.hidden_units

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

def train_model(model,trainloader,validloader):
    #validating parameters   
    learn_rate = 0.001 if args.learning_rate is None else float(args.learning_rate)
    epochs = 3 if args.epochs is None else int(args.epochs)
    device = 'cuda' if args.gpu else  'cpu'
    print("training has started")   
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    criterion = nn.NLLLoss()
    model.to(device)
    model.train()
    print_every = 40
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in  trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference<
                # Turn off gradients for validation, saves memory and computations
                model.eval()
                with torch.no_grad():    
                    test_loss, accuracy = validation(model,validloader, criterion)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                # Make sure training is back on
                model.train()
    print("trained with success !")
    save_model(model)
    return model

def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    for images,labels in validloader:
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = model.forward(images)
        test_loss += criterion(outputs,labels)
        ps = torch.exp(outputs).data
        equality = (labels == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def test(model, testloader):
    device = 'cuda' if args.gpu else  'cpu'
    model.to(device)
    test_loss = 0
    accuracy = 0

    for images,labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        ps = torch.exp(outputs).data
        equality = (labels == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    print('final accuracy on test set: {}'.format(accuracy/len(testloader)))
    


def save_model(model):
    print("saving model")
    path = 'check.pth' if args.save_dir is None else  args.save_dir
    checkpoint = {'transfer_model': 'vgg19',
                  'input_size': 25088,
                  'output_size': 102,
                  'features': model.features,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()
                 }
    torch.save(checkpoint, path)


def get_data():
    data_dir = args.data_directory
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.RandomRotation(degrees=0),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),                                  
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(batch_size=32,shuffle=True,dataset=train_datasets)
    validloader = torch.utils.data.DataLoader(batch_size=32,shuffle=True,dataset=valid_datasets)
    testloader = torch.utils.data.DataLoader(batch_size=32,shuffle=True,dataset=test_datasets)
    
    return trainloader,validloader,testloader

def main():
    global args
    args = getArgs()
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet') 
    if args.gpu  and not torch.cuda.is_available():
        raise Exception("--device gpu option enabled, but no GPU detected")
    trainloader,validloader,testloader = get_data()
    print("Starting training the model")
    model = init_transfer_model()
    model = train_model(model,trainloader,validloader)
    test(model,testloader)
    print("finished with success!")
    return None

main()