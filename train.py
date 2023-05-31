import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


# Define command-line arguments
parser = argparse.ArgumentParser(description='Train a neural network')
# Add arguments to the parser
parser.add_argument('--data_dir', type=str, default='/Users/boghdady/Desktop/aipnd-project-master/flower_data', help='Path to the directory containing the data')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='path to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden units')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--gpu', action='store_true', help='use GPU for training')

args = parser.parse_args()

def loading(data_dir):

    # Data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets["train"] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_datasets["test"] = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=6, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=6)
    test_loader = torch.utils.data.DataLoader(image_datasets["test"], batch_size=6)

    print(f"Data loaded from {data_dir} directory.")

    return image_datasets, train_loader, valid_loader, test_loader

# Define the device (GPU or CPU)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")





data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = float(args.learning_rate)
hidden_units = int(args.hidden_units)
epochs = int(args.epochs)
gpu = args.gpu

# Load and preprocess data
image_datasets, train_loader, valid_loader, test_loader = loading(data_dir)



# Build model
def build_model(arch, hidden_units):
    # Load in a pre-trained model, DenseNet default
    if arch.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088 # VGG16 input size
    elif arch.lower() == "densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 1024 # DenseNet121 input size
    else:
        raise ValueError(f"Invalid architecture choice: {arch}. Please choose between VGG16 and DenseNet121.")

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False # Freeze parameters so we don't backprop through them

    classifier = nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(0.5)),
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    print(f"Model built from {arch} and {hidden_units} hidden units.")

    return model



# Measure the validation loss and accuracy
def validation(model, dataloader, criterion, device):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in iter(dataloader):
            
            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU
            
            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output) # get the class probabilities from log-softmax
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy


# Train model
def train_model(model, train_loader, valid_loader, learning_rate, epochs, gpu):

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    device = torch.device("cuda:0" if gpu else "cpu")
    print(type(model))
    model.to(device)
    print_every = 10
    steps = 0
    running_loss = 0
    train_accuracy = 0

    print(f'Training with {learning_rate} learning rate, {epochs} epochs, and {(gpu)*"cuda" + (not gpu)*"cpu"} computing.')

    for e in range(epochs):

            model.train() # Dropout is turned on for training

            for images, labels in iter(train_loader):

                images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU

                steps += 1
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # get the class probabilities from log-softmax
                ps = torch.exp(output) 
                equality = (labels.data == ps.max(dim=1)[1])
                train_accuracy += equality.type(torch.FloatTensor).mean()

                if steps % print_every == 0:

                    model.eval() # Make sure network is in eval mode for inference

                    with torch.no_grad():
                        valid_loss, valid_accuracy = validation(model, valid_loader, criterion, device)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Training Accuracy: {:.3f}".format(train_accuracy/print_every),
                        "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                        "Validation Accuracy: {:.3f}".format(valid_accuracy/len(valid_loader)))

                    running_loss = 0
                    train_accuracy = 0
                    model.train() # Make sure training is back on
                    
    print("\nTraining completed!")
    
    return model, optimizer, criterion


# Building and training the classifier
model_init = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu)

# Save the checkpoint 
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"

print(f'Checkpoint saved to {save_dir_name}.')


