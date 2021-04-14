<p align="center"><img src='logo_github.png' /></p>

_______

FastONN - Python-based open-source GPU implementation for Operational Neural Networks

## Installation
Clone the repository and run the following command from inside the directory:
```bash
pip install .
```

## Usage
A common workflow using 2D convolutions looks as follows:
```bash
torch.nn.Conv2d(in_channels, out_channels, kernel_size)
```

This can be converted into a Self-ONN simply by swapping the convolutional layer with a Self-ONN layer:
```bash
from fastonn import SelfONNLayer
SelfONNLayer(in_channels,out_channels,kernel_size,q=3)
```
where q controls the extent of non-linearity. q=1 is equivalent to a CNN
