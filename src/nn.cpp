#include <nn.hpp>


/*
    Base shape of weights:
        rows = How many weights from previous layer connected to one neuron of the current layer
        columns = Represents a single neuron from the current layer 

*/

NN::NN(int layers, int neuronsPerLayer, int inputSize, int outputSize) {

    // ========= Initialize the input vector =============
    this->inputs = Eigen::VectorXd(inputSize);
    this->inputs.setRandom(); // Use random inputs just for now
    // ====================================================

    
    this->layers = Eigen::VectorX<Layer>(layers);
    
    //==============  Setup the first hidden layer shape ================
    Layer &firstLayer = this->layers(0);
    firstLayer.input = this->inputs;
    this->layers(0).weights.resize(neuronsPerLayer, inputSize);
    firstLayer.output = Eigen::VectorXd(neuronsPerLayer);
    firstLayer.output.setZero();
    // ==================================================================
    
    
    // ============= Setup the first layer weigths ==============
    for (int i=0; i < neuronsPerLayer; i++) 
        for (int j=0; j < inputSize; j++) 
            firstLayer.weights(i, j) = WInit::xavierInit(inputSize, neuronsPerLayer);
    // ==========================================================
    
    
    
    // ================= Initialize the input and output vectors from the rest of layers ====================
    for (int layer=1; layer < layers; layer++) {
        Layer &previousLayer = this->layers(layer-1);
        this->layers(layer).input = Eigen::VectorXd(previousLayer.output.size());
        this->layers(layer).output = Eigen::VectorXd(neuronsPerLayer);
        this->layers(layer).output.setZero();
    }
    // ======================================================================================================
    
    
    
    // ================= Initialize the weights of the rest of the weigths of the other layers and set the bias to 0 ===================
    for (int layer=1; layer < layers; layer++) {
        Layer &previousLayer = this->layers(layer - 1);
        Layer &currentLayer = this->layers(layer);
        
        currentLayer.weights.resize(neuronsPerLayer, currentLayer.input.size());
        for (int i=0; i < currentLayer.input.size(); i++)
            for (int j=0; j < neuronsPerLayer; j++)
                currentLayer.weights(i,j) = WInit::xavierInit(currentLayer.input.size(), neuronsPerLayer);
    }
    // ===========================================================================================================
    
    // ============= Initialize the output layer ================
    int lastLayerIndex = layers - 1;
    Layer &lastHiddenLayer = this->layers(lastLayerIndex);
    int outputWeightNum = lastHiddenLayer.output.size();

    output.input = Eigen::VectorXd(outputSize);
    output.output = Eigen::VectorXd(outputSize);

    output.weights.resize(outputWeightNum, outputSize);

    output.input = lastHiddenLayer.output;
    for (int i=0; i < outputWeightNum; i++)
        for (int j=0; j < outputSize; j++)
            output.weights(i, j) = WInit::xavierInit(output.input.size(), outputSize);
    // =========================================================
    
}


void NN::forward() {
    
    // ============ Weighted sum and activation on the first layer ===============
    Layer &firstLayer = this->layers(0);
    firstLayer.output = firstLayer.weights * this->inputs;

    for (auto &out : firstLayer.output)
        out = Sigmoid(out);
    //============================================================================
    
    
    for (int layer=1; layer < this->layers.size(); layer++) {
        Layer &currentLayer = this->layers(layer);
    
        currentLayer.output = currentLayer.weights * currentLayer.input;
        for (auto &out : currentLayer.output) 
            out = Sigmoid(out);
    }

    for (int layer = 0; layer < this->layers.size(); layer++) {
        std::cout << "weights of layer: " << layer << '\n';
        std::cout << this->layers(layer).weights << "\n\n";
    }
    
    /*
    int lastHiddenLayerIndex = (int) this->layers.size() - 1;
    this->output.input = this->layers(lastHiddenLayerIndex).output;
    
    
    for (int neuron=0; neuron < this->output.neurons.size(); neuron++) {
        double dotRes = this->output.neurons(neuron).weights.dot(this->output.input);
        this->output.output(neuron) = dotRes;
    }
    Softmax(this->output.output);
    */
}
