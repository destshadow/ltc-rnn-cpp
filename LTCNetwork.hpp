#pragma once

#include <vector>
#include <optional>
#include <cstddef>

#include "Vector.hpp"
#include "LTCLayer.hpp"
#include "DenseLayer.hpp"

class LTCNetwork {

private:

    std::vector<LTCLayer> layers;

    std::optional<DenseLayer>
        outputLayer;

public:

    LTCNetwork() = default;

    void addLayer(
        std::size_t inputSize,
        std::size_t hiddenSize,
        double dt
    );

    void setOutputLayer(
        std::size_t outputSize
    );

    Vector forward(
        const Vector& input
    );

    double trainStep(
        const Vector& input,
        const Vector& target,
        double learningRate
    );

    void resetState();

    std::size_t layerCount() const;

    void clearHistory();

    void resetSequence();

    double trainSequence(
        const std::vector<Vector>& sequence,
        const Vector& target,
        Optimizer& optimizer
    );

    Vector predictSequence(
        const std::vector<Vector>& sequence
    );
};
