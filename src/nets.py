import torch
import torch.nn as nn
import torch.nn.functional as F
# Refrenced from: https://github.com/vijaykumar01/stereo_matching
class Net(nn.Module):
    def __init__(self, nChannel, max_dips):
        super(Net, self).__init__()
        self.l_max_dips = max_dips
        self.conv1 = nn.Conv2d(nChannel, 32, 5)    # first conv layer: 32 filters of size 5x5
        self.batchnorm1 = nn.BatchNorm2d(32, 1e-3) # first batch normalization layer

        self.conv2 = nn.Conv2d(32, 32, 5)          # second conv layer: 32 filters of size 5x5
        self.batchnorm2 = nn.BatchNorm2d(32, 1e-3) # second normalization layer

        self.conv3 = nn.Conv2d(32, 64, 5)          # third conv layer: 64 filters of size 5x5
        self.batchnorm3 = nn.BatchNorm2d(64, 1e-3) # third batch normalization layer

        self.conv4 = nn.Conv2d(64, 64, 5)          # fourth conv layer: 64 filters of size 5x5
        self.batchnorm4 = nn.BatchNorm2d(64, 1e-3) # fourth batch normalization layer

        self.conv5 = nn.Conv2d(64, 64, 5)          # fifth conv layer: 64 filters of size 5x5
        self.batchnorm5 = nn.BatchNorm2d(64, 1e-3) # fifth batch normalization layer

        self.conv6 = nn.Conv2d(64, 64, 5)          # sixth conv layer: 64 filters of size 5x5
        self.batchnorm6 = nn.BatchNorm2d(64, 1e-3) # sixth batch normalization layer

        self.conv7 = nn.Conv2d(64, 64, 5)          # seventh conv layer: 64 filters of size 5x5
        self.batchnorm7 = nn.BatchNorm2d(64, 1e-3) # seventh batch normalization layer

        self.conv8 = nn.Conv2d(64, 64, 5)          # eighth conv layer: 64 filters of size 5x5
        self.batchnorm8 = nn.BatchNorm2d(64, 1e-3) # eigth batch normalization layer

        self.conv9 = nn.Conv2d(64, 64, 5)          # ninth conv layer: 64 filters of size 5x5
        self.batchnorm9 = nn.BatchNorm2d(64, 1e-3) # ninth batch normalization layer
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_pass(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))

        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))

        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))

        x = self.conv4(x)
        x = F.relu(self.batchnorm4(x))

        x = self.conv5(x)
        x = F.relu(self.batchnorm5(x))

        x = self.conv6(x)
        x = F.relu(self.batchnorm6(x))

        x = self.conv7(x)
        x = F.relu(self.batchnorm7(x))

        x = self.conv8(x)
        x = F.relu(self.batchnorm8(x))

        x = self.conv9(x)
        x = self.batchnorm9(x)
        return x

    def forward(self, x1, x2):
        # Forward pass left patch of 37x37
        x1 = self.forward_pass(x1)
        # Forward pass right patch of 37*237
        x2 = self.forward_pass(x2)

        # Left patch feature vector (1,64) dimension
        x1 = x1.view(x1.size(0), 1, 64)

        # Right patch feature matrix (64, 201) dimension
        # at 201 locations
        x2 = x2.squeeze().view(x2.size(0), 64, self.l_max_dips)

        # Multiply the features to get correlation at 201 location
        x3 = x1.bmm(x2).view(x2.size(0), self.l_max_dips)

        # Compute log p_i(y_i) of scores
        x3 = self.logsoftmax(x3)

        return x1, x2, x3
