#pragma once

#include <cstddef>

#include "Vector.hpp"
#include "Matrix.hpp"

class LTCLayer {

private:

    std::size_t inputSize;
    std::size_t hiddenSize;

    Matrix inputWeights;
    Matrix recurrentWeights;
    Vector bias;

    Matrix tauInputWeights;
    Matrix tauRecurrentWeights;
    Vector tauBias;

    Vector hiddenState;

    double dt;

    Vector computeTarget(
        const Vector& input
    ) const;

    Vector computeTau(
        const Vector& input
    ) const;

    Vector computeDerivative(
        const Vector& target,
        const Vector& tau
    ) const;

public:

    LTCLayer(
        std::size_t inputSize,
        std::size_t hiddenSize,
        double dt
    );

    Vector forward(
        const Vector& input
    );

    const Vector& getHiddenState() const;

    void resetState();

    void setInputWeight(
        std::size_t neuron,
        std::size_t input,
        double value
    );

    void setRecurrentWeight(
        std::size_t neuron,
        std::size_t previousNeuron,
        double value
    );

    void setBias(
        std::size_t neuron,
        double value
    );

    void setTauInputWeight(
        std::size_t neuron,
        std::size_t input,
        double value
    );

    void setTauRecurrentWeight(
        std::size_t neuron,
        std::size_t previousNeuron,
        double value
    );

    void setTauBias(
        std::size_t neuron,
        double value
    );


};