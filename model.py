import torch.nn as nn


class Judge(nn.Module):
    """
    ConvNet from TensorFlow CNN MNIST tutorial
    (see: https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier)

    Reproduced, not optmized.
    """
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(1, 32)
        self.block2 = self.conv_block(32, 64)
        self.hidden = nn.Linear(4 * 4 * 64, 1024)
        self.dropout = nn.Dropout(.4)
        self.out = nn.Linear(1024, 10)

    @classmethod
    def conv_block(cls, in_channels, out_channels):
        """A single convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )

    @classmethod
    def flatten(cls, x):
        """Flattens the input x to a 2d tensor"""
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)

    def forward(self, x):
        """
        Define a forward pass through the network.
        
        Note: Since we're inheriting from nn.Module, this will take care of the backward
        pass and parameter update step when we use an optimizer from `torch.optim`.
        """
        # make each image 3d for compatibility with convolutional blocks
        x = x.view(-1, 1, 28, 28)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x) # flatten output of convolutional section
        x = self.dropout(self.hidden(x))
        return self.out(x)
