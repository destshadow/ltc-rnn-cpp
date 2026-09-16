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
) {

    if (input.size() != inputSize) {
        throw std::invalid_argument(
            "Invalid DenseLayer input size"
        );
    }

    lastInput = input;

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

Vector DenseLayer::backward(
    const Vector& outputGradient,
    double learningRate
) {

    if (
        outputGradient.size()
        != outputSize
    ) {

        throw std::invalid_argument(
            "Invalid DenseLayer gradient size"
        );
    }

    Vector inputGradient(
        inputSize
    );

    inputGradient.fill(0.0);


    for (
        std::size_t output = 0;
        output < outputSize;
        ++output
    ) {

        for (
            std::size_t input = 0;
            input < inputSize;
            ++input
        ) {

            double weightGradient =
                outputGradient[output]
                * lastInput[input];


            inputGradient[input] +=
                weights(output, input)
                * outputGradient[output];


            weights(output, input) -=
                learningRate
                * weightGradient;
        }


        bias[output] -=
            learningRate
            * outputGradient[output];
    }


    return inputGradient;
}