#include "LTCNetwork.hpp"
#include "Loss.hpp"

#include <stdexcept>

void LTCNetwork::addLayer(
    std::size_t inputSize,
    std::size_t hiddenSize,
    double dt
) {

    if (!layers.empty()) {

        std::size_t previousOutputSize =
            layers.back().getHiddenSize();

        if (
            inputSize
            != previousOutputSize
        ) {

            throw std::invalid_argument(
                "Layer input size does not match "
                "previous layer output size"
            );
        }
    }

    layers.emplace_back(
        inputSize,
        hiddenSize,
        dt
    );
}

Vector LTCNetwork::forward(
    const Vector& input
) {

    if (layers.empty()) {

        throw std::runtime_error(
            "LTCNetwork has no layers"
        );
    }


    Vector current = input;


    for (LTCLayer& layer : layers) {

        current =
            layer.forward(current);
    }


    if (outputLayer.has_value()) {

        current =
            outputLayer->forward(current);
    }


    return current;
}

void LTCNetwork::resetState() {

    for (LTCLayer& layer : layers) {

        layer.resetState();
    }
}

std::size_t LTCNetwork::layerCount() const {

    return layers.size();
}

void LTCNetwork::setOutputLayer(
    std::size_t outputSize
) {

    if (layers.empty()) {

        throw std::runtime_error(
            "Cannot create output layer "
            "without LTC layers"
        );
    }

    std::size_t inputSize =
        layers.back().getHiddenSize();

    outputLayer.emplace(
        inputSize,
        outputSize
    );
}

double LTCNetwork::trainStep(
    const Vector& input,
    const Vector& target,
    double learningRate
) {

    if (!outputLayer.has_value()) {

        throw std::runtime_error(
            "LTCNetwork has no output layer"
        );
    }


    Vector prediction =
        forward(input);


    double loss =
        Loss::meanSquaredError(
            prediction,
            target
        );


    Vector lossGradient =
        Loss::meanSquaredErrorGradient(
            prediction,
            target
        );


    Vector gradientToLTC =
        outputLayer->backward(
            lossGradient,
            learningRate
        );

        (void)gradientToLTC;

    return loss;
}