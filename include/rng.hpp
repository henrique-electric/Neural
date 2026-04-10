#pragma once
#include <random>


/*
	The main purpose of this file is to provide a simple random number generator that can be used for initializing weights and 
    biases in the neural network. The RNG class encapsulates the functionality of generating random numbers within a specified interval 
    using the Mersenne Twister algorithm, which is a widely used pseudorandom number generator. The constructor takes two parameters, lowInterval and highInterval, 
    which define the range of the random numbers generated. The rand() method returns a random number within this range each time it is called.
*/

namespace RNG {
    struct RNG {
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> distr;
        
        RNG(double lowInterval, double highInterval) {
            gen = std::mt19937(rd());
            distr = std::uniform_real_distribution<>(lowInterval, highInterval);
        };
        
        double rand() {
            return distr(rd);
        }
    };
}

