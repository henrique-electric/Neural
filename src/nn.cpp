#include <nn.hpp>

void NN::fillMatrix(Eigen::MatrixXd &matrix, int rows, int columns) {
    matrix.resize(rows, columns);
    for (int i=0; i < rows; i++) {
        for (int j=0; j < columns; j++) {
            matrix(i, j) = WInit::xavierInit(this->inputs.size(), this->output.size());
        }
    }
}

static void fillWeigths(Eigen::VectorXd &weights, int inputSize, int outputSize) {
    for (auto &weigth : weights) {
        weigth = WInit::xavierInit(inputSize, outputSize);
    }
}

NN::NN(int layers, int neuronsPerLayer, int inputSize, int outputSize) {
    
    this->inputs = Eigen::VectorXd(inputSize);
    this->inputs.setRandom();
    
    this->output = Eigen::VectorX<Layer>(outputSize);
    
    this->layers = Eigen::VectorX<Layer>(layers);
    
    //==============  Setup the first hidden layer shape ================
    this->layers(0).input = this->inputs;
    this->layers(0).output = Eigen::VectorXd(neuronsPerLayer);
    this->layers(0).neurons = Eigen::VectorX<Neuron>(neuronsPerLayer);

    
    for (int neuron=0; neuron < neuronsPerLayer; neuron++) {
        this->layers(0).neurons(neuron).weights = Eigen::VectorXd(inputSize);
    }
    // ==================================================================
    
    
    // ============= Setup the first layer weigths ==============
    for (int neuron=0; neuron < neuronsPerLayer; neuron++) {
        int weightSize = inputSize;
        this->layers(0).neurons(neuron).weights = Eigen::VectorXd(weightSize);
        
        for (int weight=0; weight < weightSize; weight++) {
            this->layers(0).neurons(neuron).weights(weight) = WInit::xavierInit(inputSize, outputSize);
        }
    }
    // ==========================================================
    
    
    
    // ================= Initialize the input and output vectors from the rest of layers ====================
    for (int layer=1; layer < layers; layer++) {
        Layer previousLayer = this->layers(layer-1);
        this->layers(layer).input = Eigen::VectorXd(previousLayer.output.size());
        this->layers(layer).output = Eigen::VectorXd(neuronsPerLayer);
    }
    // ======================================================================================================
    
    
    // ================== Initialize the each neuron structure of the layer ======================
    for (int layer=1; layer < layers; layer++) {
        this->layers(layer).neurons = Eigen::VectorX<Neuron>(neuronsPerLayer);
    }
    // ===========================================================================================
    
    
    // ================= Initialize the weights of the rest of the weigths of the other layers ===================
    for (int layer=1; layer < layers; layer++) {
        Layer previousLayer = this->layers(layer - 1);
        Layer &currentLayer = this->layers(layer);
        
        currentLayer.input = previousLayer.output; // Set the input of the current layer as the output of the previous one
        for (int neuron=0; neuron < neuronsPerLayer; neuron++) {
            currentLayer.neurons(neuron).weights = Eigen::VectorXd(currentLayer.input.size());
            for (int weight=0; weight < neuronsPerLayer; weight++) {
                this->layers(layer).neurons(neuron).weights(weight) = WInit::xavierInit(inputSize, outputSize);
            }
        }
    }
    // ===========================================================================================================
}

// TODO
void NN::forward() {

}
