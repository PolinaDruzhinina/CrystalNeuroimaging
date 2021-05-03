import torch
import torch.nn as nn


class MriNetGrad(nn.Module):
    def __init__(self, c):
        super(MriNetGrad, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, ),

            nn.Conv3d(c, 2 * c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(2 * c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(4 * c),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1896960, out_features=2),  # 4*c*5*7*5
        )
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)
    
class MriNetGrad(nn.Module):
    def __init__(self, c):
        super(MriNetGrad, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3),
            nn.BatchNorm3d(c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3),

            nn.Conv3d(c, 2* c, kernel_size=3),
            nn.BatchNorm3d(2 * c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3),

            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(4 * c),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.MaxPool3d(kernel_size=3),
            nn.Flatten(),
            #             nn.Linear(in_features=4*c*5*5*5, out_features=2),
            nn.Linear(in_features=43008, out_features=2)
        )
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)


hidden = lambda c_in, c_out: nn.Sequential(
    nn.Conv3d(c_in, c_out, (3, 3, 3)),  # Convolutional layer
    nn.BatchNorm3d(c_out),  # Batch Normalization layer
    nn.ReLU(),  # Activational layer
    nn.MaxPool3d(3)  # Pooling layer
)


class MriNet(nn.Module):
    def __init__(self, c):
        super(MriNet, self).__init__()
        self.hidden1 = hidden(1, c)
        self.hidden2 = hidden(c, 2 * c)
        self.hidden3 = hidden(2 * c, 4 * c)
        self.hidden4 = hidden(4 * c, 4 * c)
        self.linear = nn.Linear(256, 2)  # 16000
        #         self.linear = nn.Linear(128*5*5*5, 2)#16000
        self.flatten = nn.Flatten()

    def forward(self, x):
        #         print(x.shape)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        #         print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)

        x = F.log_softmax(x, dim=1)
        return x
