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
from fastonn import SelfONN2dLayer
SelfONN2dLayer(in_channels, out_channels, kernel_size, q=3)
```
where q controls the extent of non-linearity. q=1 is equivalent to a CNN

For using super neurons, the additional parameter of max_shift is required

```bash
from fastonn import SuperONN2dLayer
SuperONN2dLayer(in_channels, out_channels, kernel_size, max_shift=18, q=3)
```


## Precautions
- A SelfONN2dLayer expects input to be bounded between [-1,1]. To respect this, a Tanh or Sigmoid activation layer must preceed it.
- For SuperONN2dLayer, the max_shift must respect the size of the input feature map. The layer does not raise a flag if the shift exceeds the boundaries of the image.

## References
- Kiranyaz, S, Ince, T, Iosifidis, A & Gabbouj, M 2020, 'Operational neural networks', Neural Computing and Applications, vol. 32, pp. 6645–6668. https://doi.org/10.1007/s00521-020-04780-3
- Serkan Kiranyaz, Junaid Malik, Habib Ben Abdallah, Turker Ince, Alexandros Iosifidis, Moncef Gabbouj, Self-organized Operational Neural Networks with Generative Neurons, Neural Networks, Volume 140, 2021, Pages 294-308, ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2021.02.028.
- Junaid Malik, Serkan Kiranyaz, Moncef Gabbouj,
Self-organized operational neural networks for severe image restoration problems, Neural Networks, Volume 135, 2021, Pages 201-211, ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2020.12.014.
