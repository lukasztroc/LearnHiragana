from PIL import Image
import PIL.ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision.models import resnet18, resnet34
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import Dataset

class KMNIST(Dataset):

    def __init__(self, train=True, root=''):
        self.root = root

        self.X_train = torch.from_numpy(
            np.load(root + "hiragana-20210405T130133Z-001/hiragana/k49-train-imgs.npz")['arr_0']).reshape(-1, 1, 28, 28)
        self.X_train = self.X_train / 255
        self.y_train = torch.from_numpy(
            np.load(root + "hiragana-20210405T130133Z-001/hiragana/k49-train-labels.npz")['arr_0'])
        self.y_train = self.y_train.type(torch.LongTensor)

        self.X_test = torch.from_numpy(
            np.load(root + "hiragana-20210405T130133Z-001/hiragana/k49-test-imgs.npz")['arr_0']).reshape(-1, 1, 28, 28)
        self.X_test = self.X_test / 255
        self.y_test = torch.from_numpy(
            np.load(root + "hiragana-20210405T130133Z-001/hiragana/k49-test-labels.npz")['arr_0'])
        self.y_test = self.y_test.type(torch.LongTensor)

        self.train = train

    def __len__(self):
        return self.X_train.shape[0] if self.train else self.X_test.shape[0]

    def __getitem__(self, index):
        if self.train:
            return self.X_train[index], self.y_train[index]

        else:
            return self.X_test[index], self.y_test[index]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 49)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output


class ResNetKMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=49)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def img_preprocess(image):
    image = PIL.ImageOps.invert(PIL.ImageOps.grayscale(image.resize((28, 28))))
    return np.array(image).reshape((-1, 1, 28, 28)) / 255


def load_model(path):
    net = torch.load(path, map_location=torch.device('cpu'))
    return net.double()


def load_model_from_dict(path):
    m_state_dict = torch.load(path, map_location=torch.device('cpu'))
    inference_model = ResNetKMNIST()
    inference_model.load_state_dict(m_state_dict)
    return inference_model.double()


def classify(net, image):
    net.eval()
    tensor = torch.from_numpy(img_preprocess(image))
    with torch.no_grad():
        pred = net(tensor)
        a = F.softmax(pred, dim=1)
        return int(torch.argmax(pred)), float(a.max())


def get_label(index):
    df = pd.read_csv('files/mappings.csv')
    asd = list(df.where(df['index'] == index).dropna()['phonetic'])
    return asd[0]


def get_sample(index):
    return Image.open(f'files/samples/{index}.png')
