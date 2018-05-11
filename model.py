import torch.nn as nn


class Judge(nn.Module):
    """
    ConvNet from TensorFlow CNN MNIST tutorial
    (see: https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier)

    Reproduced, not optimized.  Only change is number of channels in input.
    """
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(2, 32)
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
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)

    def forward(self, x):
        """Forward pass"""
        x = x.view(-1, 2, 28, 28)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dropout(self.hidden(x))
        return self.out(x)
