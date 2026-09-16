#pragma once

#include <cstddef>

#include "Vector.hpp"
#include "Matrix.hpp"
#include <vector>

struct LTCStepCache {

    Vector input;

    Vector previousState;

    Vector target;

    Vector tau;

    Vector derivative;
};

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

    std::vector<LTCStepCache> history;

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

    double clipGradient(
        double gradient,
        double limit
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

    std::size_t getInputSize() const;
    std::size_t getHiddenSize() const;

    void clearHistory();
    std::size_t getHistorySize() const;

    std::vector<Vector> backward(
        const std::vector<Vector>& outputGradients,
        double learningRate
    );


};
