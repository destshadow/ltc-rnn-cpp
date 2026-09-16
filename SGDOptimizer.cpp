#include "SGDOptimizer.hpp"

#include <stdexcept>


SGDOptimizer::SGDOptimizer(
    double learningRate
)
    : learningRate(learningRate) {

    if (learningRate <= 0.0) {

        throw std::invalid_argument(
            "learningRate must be greater than zero"
        );
    }
}


double SGDOptimizer::update(
    double parameter,
    double gradient
) {

    return parameter
        - learningRate * gradient;
}


double SGDOptimizer::getLearningRate() const {

    return learningRate;
}