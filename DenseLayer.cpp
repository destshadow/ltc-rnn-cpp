#include "DenseLayer.hpp"

#include "Initializer.hpp"

#include <stdexcept>


DenseLayer::DenseLayer(
    std::size_t inputSize,
    std::size_t outputSize
)
    : inputSize(inputSize),
      outputSize(outputSize),
      weights(outputSize, inputSize),
      bias(outputSize) {

    Initializer::randomize(
        weights,
        -0.5,
        0.5
    );

    bias.fill(0.0);
}

Vector DenseLayer::forward(
    const Vector& input
) const {

    if (input.size() != inputSize) {

        throw std::invalid_argument(
            "Invalid DenseLayer input size"
        );
    }

    return weights * input + bias;
}

std::size_t DenseLayer::getInputSize() const {

    return inputSize;
}


std::size_t DenseLayer::getOutputSize() const {

    return outputSize;
}

void DenseLayer::setWeight(
    std::size_t output,
    std::size_t input,
    double value
) {

    weights(
        output,
        input
    ) = value;
}

void DenseLayer::setBias(
    std::size_t output,
    double value
) {

    bias[output] = value;
}