#include <nn.hpp>


double NN::Sigmoid(double x) {
    double denominator = 1 + pow(Constants::euler, -x);
    return 1/denominator;
}

double NN::reLU(double x) {
    if (x <= 0)
        return 0;
    else
        return x;
}

void NN::Softmax(Eigen::VectorXd &vec) {
    double sum = 0;
    for (auto &x : vec)
        sum += pow(Constants::euler, x);
    
    for (auto &x : vec) {
        x = pow(Constants::euler, x) / sum;
    }
}

void NN::GradientCalc(void) {
    int lastLayerindex = (int) this->layers.size();
    double newWeight = 0;
    
    Layer &outputLayer = this->output;
    int outputLayerSize = (int) outputLayer.output.size();
    
    for (int output=0; output < outputLayerSize; output++) {
        double chageToA = Loss::SquareLossDerivative(outputLayer.output(output), this->trueLabels(output));
        std::cout << chageToA;
    }
    
    /*
    
    for (int layer=lastLayerindex; layer > 0; layer--) {
        Layer &currentLayer = this->layers(layer);
        
        
        for (int neuron=0; neuron < currentLayer.neurons.size(); neuron++) {
            for (auto &weight : currentLayer.neurons(neuron).weights) {
                
            }
        }
    }
     */
}

// Marked to remove
void NN::activation(Eigen::VectorXd &linearOutput) {
    for (auto &value : linearOutput) {
        value = NN::Sigmoid(value);
    }
}
