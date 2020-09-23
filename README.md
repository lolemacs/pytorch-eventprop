# Spiking Neural Network training with EventProp

This is an unofficial PyTorch implemenation of [EventProp](https://arxiv.org/abs/2009.08378), a method to compute exact gradients for Spiking Neural Networks. The repo currently contains code to train a 1-layer Spiking Neural Network with leaky integrate-and-fire (LIF) neurons for 10-way digit classification on MNIST.

## Implementation Details

The implementation of EventProp itself is in *models.py*, in form of the forward and backward methods of the SpikingLinear module, which compute the forward passes of a spiking layer and its adjoint layer.

In particular, the *manual_forward* method computes the discretized dynamics of a spiking layer:

<img src="https://github.com/lolemacs/pytorch-eventprop/blob/master/images/forward.png" height="180">

While the *manual_backward* method computes the discretized dynamics of the adjoint model, used to compute exact gradients for the weight parameters:

<img src="https://github.com/lolemacs/pytorch-eventprop/blob/master/images/backward.png" height="180">


The network is run for a fixed amount of time and discrete time steps are used to approximate the continuous dynamics. These can be set through the *T* and *dt* arguments when running *main.py* (default values are T=40ms and dt=1ms, so a total of 40 forward passes are executed for each mini-batch).

To encode the MNIST dataset as spikes, images were first binarized and black/white pixels were encoded as spikes at times 10/20ms, respectively. The dynamics of one of the 10 output neurons are as follows, for a randomly-initialized network:

<img src="https://github.com/lolemacs/pytorch-eventprop/blob/master/images/simulation.png" width="800">

where vertical black lines indicate spike times.

## Usage

The code was tested with Python 2.7 + PyTorch 0.4 and Python 3.8 + PyTorch 1.4, producing similar results.

To train the SNN with default settings, just run
```
python main.py
```
which will automatically download MNIST and train a SNN for 40 epochs with Adam, on gpu.

Check out the available args in *main.py* to change training settings such as the learning rate, batch size, and SNN-specific parameters such as membrane/synaptic constants and time discretization.

The default hyperparameters result in stable training, reaching around 85% train/test accuracy in under 10 epochs:

<img src="https://github.com/lolemacs/pytorch-eventprop/blob/master/images/eventprop.png" width="800">

## Extensions

If there is enough interest, I can try to extend the EventProp implementation to handle hidden layers / convolutions. If you'd like to extend it yourself, feel free to submit a pull request.
