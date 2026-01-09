#include <nn.hpp>


/*
    Base shape of weights:
        rows = How many weights from previous layer connected to one neuron of the current layer
        columns = Represents a single neuron from the current layer

*/

NN::NN(int layers, int neuronsPerLayer, int inputSize, int outputSize, Loss::LossOptions loss, OutputActivation outLayerActivation,
       HiddenActivation hiddenActivation) {

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
        
        // rows = neuronsPerLayer, cols = size of input to this layer
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

    output.input = Eigen::VectorXd(outputWeightNum);
    output.output = Eigen::VectorXd(outputSize);

    output.weights.resize(outputWeightNum, outputSize);

    output.input = lastHiddenLayer.output;
    for (int i=0; i < outputWeightNum; i++)
        for (int j=0; j < outputSize; j++)
            output.weights(i, j) = WInit::xavierInit(output.input.size(), outputSize);
    // =========================================================
    
    this->outputLayerActivation = outLayerActivation;
    this->loss = loss;
    // store the chosen hidden activation (was previously omitted)
    this->hiddenLayerActivation = hiddenActivation;
}

void NN::handleOutActivation(void) {
    Layer &outputLayer = this->output;

    switch (this->outputLayerActivation)
    {
    case SIGMOID_OUT:
        for (auto &out : outputLayer.output)
            out = Sigmoid(out);
        return;

    case SOFTMAX:
        Softmax(outputLayer.output);
        return;
    
    default:
        break;
    }
}

double NN::handleActivation(double out)
{
    switch (this->hiddenLayerActivation)
    {
    case SIGMOID:
        return Sigmoid(out);

    case RELU:
        return reLU(out);
    
    default:
        break;
    }
}

void NN::forward() {
    
    // ============ Weighted sum and activation on the first layer ===============
    Layer &firstLayer = this->layers(0);
    firstLayer.output = firstLayer.weights * this->inputs;

    for (auto &out : firstLayer.output) {
        out = Sigmoid(out);
    }
    //============================================================================
    
    
    for (int layer=1; layer < this->layers.size(); layer++) {
        Layer &previousLayer = this->layers(layer - 1);
        Layer &currentLayer = this->layers(layer);
        

        currentLayer.input = previousLayer.output;
        currentLayer.output = currentLayer.weights * currentLayer.input;

        for (auto &out : currentLayer.output) {
            out = handleActivation(out);
        }
            
    }
    
    int lastHiddenLayerIndex = this->layers.size() - 1;
    Layer &lastHiddenLayer = this->layers(lastHiddenLayerIndex);

    this->output.input = lastHiddenLayer.output;
    std::cout << this->output.input.size() << '\n';
    std::cout << lastHiddenLayer.output.size() << '\n';
   
    for (auto &out : this->output.output)
        out = Sigmoid(out);
}


#ifdef COMPILE_UTILS
void NN::printLayerWeights(void) {
    for (int layer=0; layer < this->layers.size(); layer++) {
        std::cout << "Weights from layer " << layer << '\n';
        std::cout << this->layers(layer).weights << "\n\n";
    }
}

void NN::printLayerOutputs(void) {
    for (int layer=0; layer < this->layers.size(); layer++) {
        std::cout << "Output from layer " << layer << '\n';
        std::cout << this->layers(layer).output << "\n\n";
    }
}

void NN::printLayerInputs(void) {
    for (int layer=0; layer < this->layers.size(); layer++) {
        std::cout << "Inputs of layer " << layer << '\n';
        std::cout << this->layers(layer).input << "\n\n";
    }
}

void NN::printOutputLayerResult(void) {
    std::cout << "Output vector of output layer\n";
    std::cout << this->output.output << "\n\n";
}

void NN::printOuputLayerInput(void) {
    std::cout << "Input vector of output layer\n";
    std::cout << this->output.input << "\n\n";
}

#endif
