#pragma once
#include <random>

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

