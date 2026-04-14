#include <nn.hpp>

#ifdef COMPILE_UTILS
void NN::printLayerWeights(void) {
    for (int layer=0; layer < this->layers.size(); layer++) {
        std::cout << "Weights from layer " << layer << '\n';
        std::cout << this->layers(layer).weights << "\n\n";
    }
}

void NN::printLayerOutputs(void) {
    for (int layer = 0; layer < this->layers.size(); layer++) {
        std::cout << "Output from layer " << layer << '\n';
        std::cout << this->layers(layer).output << "\n";
    }
}

void NN::printLayerInputs(void) {
    for (int layer = 0; layer < this->layers.size(); layer++) {
        std::cout << "Input from layer " << layer << '\n';
        std::cout << this->layers(layer).input << "\n";
    }
}
#endif