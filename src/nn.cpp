#include <nn.hpp>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> distr(-5, 5);


static void fillMatrix(Eigen::MatrixXd &matrix, int rows, int columns) {
    for (int i=0; i < rows; i++) {
        for (int j=0; j < columns; j++) {
            matrix(i, j) = distr(rd);
        }
    }
}

NN::NN(int layers, int neuronsPerLayer){
    this->inputs = Eigen::VectorXd(neuronsPerLayer);
    this->inputs.setRandom();
    
    this->layers = Eigen::VectorX<Layer>(layers);
    
    for (int i=0; i < this->layers.size(); i++) {
        this->layers(i).weights.resize(layers, neuronsPerLayer);
        fillMatrix(this->layers(i).weights, layers, neuronsPerLayer);
    }
        
       
}

// TODO
void NN::forward() {
    Eigen::VectorXd linearOutput;
    //std::cout << this->layers(0).weights;
    //std::cout << "\n\n\n";
    
    Eigen::VectorXd ws = this->layers(0).weights * this->inputs;
    this->activation(ws);
}
