#include <loss.hpp>
#include <cmath>


double Loss::squareLoss(double result, double expected) {
    return pow(result - expected, 2);
}
