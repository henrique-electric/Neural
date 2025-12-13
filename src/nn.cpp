#include <nn.hpp>

NN::NN(int layers, int neuronsPerLayer, int inputSize, int outputSize) {

    // ========= Initialize the input vector =============
    this->inputs = Eigen::VectorXd(inputSize);
    this->inputs.setRandom(); // Use random inputs just for now
    // ====================================================

    
    // ======== Initialize the layer vector and get references of the first Hidden layer and the output layer =======
    this->layers = Eigen::VectorX<Layer>(layers);
    Layer &outputLayer = this->output;
    Layer &firstLayer = this->layers(0);
    //================================================================================================================
    
    
    //==============  Setup the first hidden layer shape ================
    firstLayer.input = this->inputs;
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
            this->layers(0).neurons(neuron).weights(weight) = WInit::xavierInit(this->inputs.size(), this->layers(0).output.size());
        }
    }
    // ==========================================================
    
    
    
    // ================= Initialize the input and output vectors from the rest of layers ====================
    for (int layer=1; layer < layers; layer++) {
        Layer &previousLayer = this->layers(layer-1);
        this->layers(layer).input = Eigen::VectorXd(previousLayer.output.size());
        this->layers(layer).output = Eigen::VectorXd(neuronsPerLayer);
        this->layers(layer).output.setZero();
    }
    // ======================================================================================================
    
    
    // ================== Initialize the each neuron structure of the layer ======================
    for (int layer=1; layer < layers; layer++) {
        this->layers(layer).neurons = Eigen::VectorX<Neuron>(neuronsPerLayer);
    }
    // ===========================================================================================
    
    
    // ================= Initialize the weights of the rest of the weigths of the other layers and set the bias to 0 ===================
    for (int layer=1; layer < layers; layer++) {
        Layer &previousLayer = this->layers(layer - 1);
        Layer &currentLayer = this->layers(layer);
        
        currentLayer.input = previousLayer.output; // Set the input of the current layer as the output of the previous one
        for (int neuron=0; neuron < neuronsPerLayer; neuron++) {
            currentLayer.neurons(neuron).weights = Eigen::VectorXd(currentLayer.input.size());
            currentLayer.neurons(neuron).bias = 0;
            
            for (int weight=0; weight < currentLayer.input.size(); weight++) {
                this->layers(layer).neurons(neuron).weights(weight) = WInit::xavierInit(previousLayer.output.size(), currentLayer.output.size());
            }
        }
    }
    // ===========================================================================================================
    
    // ============= Initialize the output layer ================
    outputLayer.neurons = Eigen::VectorX<Neuron>(outputSize);
    outputLayer.input = Eigen::VectorXd(outputSize);
    outputLayer.output = Eigen::VectorXd(outputSize);
    
    int lastLayerIndex = layers - 1;
    Layer &lastHiddenLayer = this->layers(lastLayerIndex);
    outputLayer.input = lastHiddenLayer.output;
    for (int neuron=0; neuron < outputSize; neuron++) {
        outputLayer.neurons(neuron).weights = Eigen::VectorXd(outputLayer.input.size());
        
        for (auto &weight : outputLayer.neurons(neuron).weights) {
            weight = WInit::xavierInit(outputLayer.input.size(), outputLayer.output.size());
        }
    }
    // =========================================================
    
}


void NN::forward() {
    
    Layer &firstLayer = this->layers(0);
    for (int neuron=0; neuron < firstLayer.neurons.size(); neuron++) {
        double dotRes = firstLayer.neurons(neuron).weights.dot(firstLayer.input);
        firstLayer.output(neuron) = Sigmoid(dotRes);

    }
    
    for (int layer=1; layer < this->layers.size(); layer++) {
        this->layers(layer).input = this->layers(layer - 1).output;
        Eigen::VectorXd &currentInput = this->layers(layer).input;
        
        for (int neuron=0; neuron < this->layers(layer).neurons.size(); neuron++) {
            double dotRes = this->layers(layer).neurons(neuron).weights.dot(currentInput) + this->layers(layer).neurons(neuron).bias;
            this->layers(layer).output(neuron) = Sigmoid(dotRes);
        }
    }
    
    int lastHiddenLayerIndex = (int) this->layers.size() - 1;
    this->output.input = this->layers(lastHiddenLayerIndex).output;
    
    
    for (int neuron=0; neuron < this->output.neurons.size(); neuron++) {
        double dotRes = this->output.neurons(neuron).weights.dot(this->output.input);
        this->output.output(neuron) = dotRes;
    }
    Softmax(this->output.output);
}
