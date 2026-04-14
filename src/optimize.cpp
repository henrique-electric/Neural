#include <optimize.hpp>

void NN::GradientCalc() {
	int currentLayerIndex = this->layers.size() - 1;
	int numberOfLayers = this->layers.size();
	auto& currentLayer = this->layers(currentLayerIndex);
	auto& outputLayer = this->output;

	double lossOut = 0;


	// The gradient vector for each 
	Eigen::VectorXd outputLayerGradient(outputLayer.output.size());
	outputLayerGradient.setZero();

	for (int gradientIndex = 0; gradientIndex < outputLayerGradient.size(); gradientIndex++) {
		int outputIndex = gradientIndex;
		int trueLabelIndex = gradientIndex;

		
		
	}


	/*
		Create a matrix to hold all the gradient vectors for backpropagatio for the hidden layers;
		It will be a matrix since it will hold a gradien vector for each hidden layer.

		Shape:
			column= a gradient vector of a layer
			row= a partial derivative of a the loss for that neuron
	*/
	Eigen::MatrixXd gradientMatrix(numNeuronsPerLayer, layers);

	/*
		Considering the shape of the matrix of weights on "nn.cpp", we need the for loop to be
		in this order, since the each row represent the weights of a single neuron on the layer
		in this case, we'll be tweaking the weights of each neuron on the layer.
	*/
	for (int column = 0; column < currentLayer.weights.cols(); i++) {
		for (int row = 0; row < currentLayer.weights.rows(); row++) {

		}
	}


}