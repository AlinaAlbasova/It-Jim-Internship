import json
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from PIL import Image

TRAIN_DIR = "splitted_dataset/train/"
TEST_DIR = "splitted_dataset/test/"
VAL_DIR = "splitted_dataset/val/"

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
val_images = [VAL_DIR+i for i in os.listdir(VAL_DIR)]
test_images =[TEST_DIR+i for i in os.listdir(TEST_DIR)]

classes = os.listdir("dataset/")
class_numbers = range(0,16)

device = torch.device('cpu')

from data_preparation import prep_data
from models_eval import eval


def load_model(PATH):
    model = torch.load(PATH)
    return model


def save_model(model, PATH):
    torch.save(model, PATH)
    print("Model is saved.")


def train_nn(nn_class, trainloader):
    net = nn_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if nn_class == CNN:
                inputs = np.transpose(inputs, (0, 3, 1, 2))

            labels = labels.type(torch.LongTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))

            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    return net


def test(net, net_class, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if net_class == 'CNN':
                images = np.transpose(images, (0, 3, 1, 2))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            eval(labels, predicted)


    # print('Accuracy of the network on the test images: %d %%' % (
    #         100 * correct / total))


def infer(path_to, model, mode='image'):
    if mode == 'image':
        image = Image.open(path_to)
        transformation = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
        image = transformation(image).float() / 255.0
        image = image.unsqueeze(0)

        pred_classes = model(image)
        pred_classes = torch.squeeze(pred_classes)
        pred_classes = pred_classes.tolist()
        pred_class = np.argmax(pred_classes)

        pred_class = classes[pred_class]

        np.set_printoptions(precision=3, suppress=True)
        for i in range(16):
            print(classes[i], " - ", pred_classes[i])
        print("Best match for ", path_to, " is ", pred_class)

    elif mode == 'folder':
        folder_predictions = {}
        for fn in os.listdir(path_to):
            # predicting images
            path = path_to + fn
            image = Image.open(path_to)
            transformation = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
            image = transformation(image).float() / 255.0
            image = image.unsqueeze(0)

            pred_classes = model(image)
            pred_classes = torch.squeeze(pred_classes)
            pred_classes = pred_classes.tolist()
            pred_class = np.argmax(pred_classes)

            pred_class = classes[pred_class]

            np.set_printoptions(precision=3, suppress=True)

            predictions_dict = {}

            np.set_printoptions(precision=3, suppress=True)
            for i in range(16):
                predictions_dict[classes[i]] = str(pred_classes[i])
                print(classes[i], " - ", pred_classes[i])

            folder_predictions[fn] = predictions_dict

            with open('predictions_torch.json', 'w') as fp:
                json.dump(folder_predictions, fp)

            print("Best match for ", fn, " is ", pred_class)


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(84 * 84 * 3, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, 16)

    def forward(self, x):
        x = x.view(-1, 84 * 84 * 3)
        x = F.relu((self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20*18*18, 50)
        self.fc2 = nn.Linear(50, 16)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

if __name__ == '__main__':

    X_train, Y_train = prep_data(train_images)
    X_train = X_train / 255.0
    X_train_tensor, Y_train_tensor = torch.Tensor(X_train), torch.Tensor(Y_train)

    X_val, Y_val = prep_data(val_images)
    X_val_tensor, Y_val_tensor = torch.Tensor(X_val), torch.Tensor(Y_val)

    X_test, Y_test = prep_data(test_images)
    X_test = X_test / 255.0
    X_test_tensor, Y_test_tensor = torch.Tensor(X_test), torch.Tensor(Y_test)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=240, shuffle=False)

    nett = train_nn(FCNN, train_dataloader)
    test(nett, 'FCNN', test_dataloader)
    path = "inf3.jpg"
    path2 = "splitted_dataset/val/"
    infer(path, nett)