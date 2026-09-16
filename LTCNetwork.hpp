#pragma once

#include <vector>
#include <cstddef>

#include "Vector.hpp"
#include "LTCLayer.hpp"

class LTCNetwork {

private:

    std::vector<LTCLayer> layers;

public:

    LTCNetwork() = default;

    void addLayer(
        std::size_t inputSize,
        std::size_t hiddenSize,
        double dt
    );

    Vector forward(
        const Vector& input
    );

    void resetState();

    std::size_t layerCount() const;
};