#pragma once

#include <cstddef>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"

class DenseLayer {

private:

    std::size_t inputSize;
    std::size_t outputSize;

    Matrix weights;
    Vector bias;

    Vector lastInput;

public:

    DenseLayer(
        std::size_t inputSize,
        std::size_t outputSize
    );

    Vector forward(
        const Vector& input
    );

    Vector backward(
        const Vector& outputGradient,
        Optimizer& optimizer
    );

    std::size_t getInputSize() const;
    std::size_t getOutputSize() const;

    void setWeight(
        std::size_t output,
        std::size_t input,
        double value
    );

    void setBias(
        std::size_t output,
        double value
    );
};
