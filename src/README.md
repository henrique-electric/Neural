## A guide for what each file holds on this source folder

- `optimize.cpp`: Contains the functions and the core for optimization of the network, gradient calculation, backpropagation etc...
- `nn.cpp`: The core of the network, this is where functions that create the network itself and setup the shape of the network and the 'forward' function that feeds the network.
- `loss.cpp`: Contains the definitions for the loss functions + their derivatives.
- `activations.cpp`: Contains the activation functions available on the network + their derivatives.
- `inits.cpp`: The functions initialize the weights and biases.
- `compile_utils.cpp`: Theses functions are only useful for debugging or to have a more visual and verbose status of the network.
