import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, channels, scale_factor, f1 = 9, f2 = 1, f3 = 5, n1 = 64, n2 = 32):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(channels, n1, f1, padding=f1//2)
        self.conv2 = nn.Conv2d(n1, n2, f2, padding=f2//2)
        self.conv3 = nn.Conv2d(n2, channels, f3, padding=f3//2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)
