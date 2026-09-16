#pragma once

#include <cstddef>

#include "Vector.hpp"
#include "Matrix.hpp"

class RNNLayer {

private:

    std::size_t inputSize;
    std::size_t hiddenSize;

    Matrix inputWeights;
    Matrix recurrentWeights;

    Vector bias;
    Vector hiddenState;

public:

    RNNLayer(
        std::size_t inputSize,
        std::size_t hiddenSize
    );

    Vector forward(const Vector& input);

    const Vector& getHiddenState() const;

    void resetState();
    
    void setInputWeight(
    std::size_t row,
    std::size_t col,
    double value
);

void setRecurrentWeight(
    std::size_t row,
    std::size_t col,
    double value
);

void setBias(
    std::size_t index,
    double value
);
};

