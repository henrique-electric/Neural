#include <nn.hpp>

double NN::Sigmoid(double x) {
    double denominator = 1 + pow(Constants::euler, -x);
    return 1/denominator;
}

NN::NN(int layers, int neuronsPerLayer){
    this->inputs = Eigen::VectorX<double>(neuronsPerLayer);
    this->layers = Eigen::VectorX<Layer>(layers);
    
    for (int i=0; i < this->layers.size(); i++) {
        this->layers(i).weights.resize(layers, neuronsPerLayer);
        this->layers(i).weights.setRandom();
    }
        
       
}

void NN::forward() {
    double val = this->layers(1,1).weights(1,2);
    std::cout << this->layers(1,1).weights.size();
}
