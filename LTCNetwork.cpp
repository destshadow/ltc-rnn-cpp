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

void LTCNetwork::clearHistory() {

    for (LTCLayer& layer : layers) {

        layer.clearHistory();
    }
}

void LTCNetwork::resetSequence() {

    resetState();
    clearHistory();
}

double LTCNetwork::trainSequence(
    const std::vector<Vector>& sequence,
    const Vector& target,
    double learningRate
) {

    if (layers.empty()) {

        throw std::runtime_error(
            "LTCNetwork has no LTC layers"
        );
    }

    if (!outputLayer.has_value()) {

        throw std::runtime_error(
            "LTCNetwork has no output layer"
        );
    }

    if (sequence.empty()) {

        throw std::invalid_argument(
            "Cannot train on empty sequence"
        );
    }


    resetSequence();


    Vector finalHiddenState;


    for (
        const Vector& input : sequence
    ) {

        Vector current =
            input;

        for (
            LTCLayer& layer : layers
        ) {

            current =
                layer.forward(current);
        }

        finalHiddenState =
            current;
    }


    Vector prediction =
        outputLayer->forward(
            finalHiddenState
        );


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


    Vector finalLTCGradient =
        outputLayer->backward(
            lossGradient,
            learningRate
        );


    std::vector<Vector> timeGradients;

    timeGradients.reserve(
        sequence.size()
    );


    std::size_t lastHiddenSize =
        layers.back().getHiddenSize();


    for (
        std::size_t t = 0;
        t < sequence.size();
        ++t
    ) {

        Vector gradient(
            lastHiddenSize
        );

        gradient.fill(
            0.0
        );

        timeGradients.push_back(
            gradient
        );
    }


    timeGradients.back() =
        finalLTCGradient;


    for (
        std::size_t layerIndex =
            layers.size();
        layerIndex-- > 0;
    ) {

        timeGradients =
            layers[layerIndex].backward(
                timeGradients,
                learningRate
            );
    }


    return loss;
}

Vector LTCNetwork::predictSequence(
    const std::vector<Vector>& sequence
) {

    if (layers.empty()) {

        throw std::runtime_error(
            "LTCNetwork has no LTC layers"
        );
    }

    if (!outputLayer.has_value()) {

        throw std::runtime_error(
            "LTCNetwork has no output layer"
        );
    }

    if (sequence.empty()) {

        throw std::invalid_argument(
            "Cannot predict empty sequence"
        );
    }


    resetSequence();


    Vector current;


    for (
        const Vector& input : sequence
    ) {

        current =
            input;

        for (
            LTCLayer& layer : layers
        ) {

            current =
                layer.forward(current);
        }
    }


    return outputLayer->forward(
        current
    );
}