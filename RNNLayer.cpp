#include "RNNLayer.hpp"

#include "Activation.hpp"

#include <stdexcept>

RNNLayer::RNNLayer(
    std::size_t inputSize,
    std::size_t hiddenSize
)
    : inputSize(inputSize),
      hiddenSize(hiddenSize),
      inputWeights(hiddenSize, inputSize),
      recurrentWeights(hiddenSize, hiddenSize),
      bias(hiddenSize),
      hiddenState(hiddenSize) {
}

void RNNLayer::setInputWeight(
    std::size_t row,
    std::size_t col,
    double value
) {

    inputWeights(row, col) = value;
}

void RNNLayer::setRecurrentWeight(
    std::size_t row,
    std::size_t col,
    double value
) {

    recurrentWeights(row, col) = value;
}

void RNNLayer::setBias(
    std::size_t index,
    double value
) {

    bias[index] = value;
}

Vector RNNLayer::forward(
    const Vector& input
) {

    if (input.size() != inputSize) {
        throw std::invalid_argument(
            "Invalid RNN input size"
        );
    }


    Vector inputContribution =
        inputWeights * input;


    Vector memoryContribution =
        recurrentWeights * hiddenState;


    Vector z =
        inputContribution
        + memoryContribution
        + bias;


    hiddenState =
        Activation::tanh(z);


    return hiddenState;
}

const Vector& RNNLayer::getHiddenState() const {

    return hiddenState;
}

void RNNLayer::resetState() {

    hiddenState.fill(0.0);
}