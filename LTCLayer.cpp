#include "LTCLayer.hpp"

#include "Activation.hpp"

#include <stdexcept>


LTCLayer::LTCLayer(
    std::size_t inputSize,
    std::size_t hiddenSize,
    double dt
)
    : inputSize(inputSize),
      hiddenSize(hiddenSize),

      inputWeights(
          hiddenSize,
          inputSize
      ),

      recurrentWeights(
          hiddenSize,
          hiddenSize
      ),

      bias(hiddenSize),

      tauInputWeights(
          hiddenSize,
          inputSize
      ),

      tauRecurrentWeights(
          hiddenSize,
          hiddenSize
      ),

      tauBias(hiddenSize),

      hiddenState(hiddenSize),

      dt(dt) {
}

Vector LTCLayer::computeTarget(
    const Vector& input
) const {

    Vector inputContribution =
        inputWeights * input;

    Vector recurrentContribution =
        recurrentWeights * hiddenState;

    Vector combined =
        inputContribution
        + recurrentContribution
        + bias;

    return Activation::tanh(
        combined
    );
}

Vector LTCLayer::computeTau(
    const Vector& input
) const {

    Vector inputContribution =
        tauInputWeights * input;

    Vector recurrentContribution =
        tauRecurrentWeights
        * hiddenState;

    Vector combined =
        inputContribution
        + recurrentContribution
        + tauBias;

    Vector sigmoidValues =
        Activation::sigmoid(
            combined
        );

    Vector tau(hiddenSize);

    for (
        std::size_t i = 0;
        i < hiddenSize;
        ++i
    ) {

        tau[i] =
            1.0 + sigmoidValues[i];
    }

    return tau;
}

Vector LTCLayer::computeDerivative(
    const Vector& target,
    const Vector& tau
) const {

    Vector derivative(
        hiddenSize
    );

    for (
        std::size_t i = 0;
        i < hiddenSize;
        ++i
    ) {

        derivative[i] =
            (
                target[i]
                - hiddenState[i]
            )
            / tau[i];
    }

    return derivative;
}

Vector LTCLayer::forward(
    const Vector& input
) {

    if (
        input.size()
        != inputSize
    ) {

        throw std::invalid_argument(
            "Invalid LTC input size"
        );
    }


    Vector target =
        computeTarget(input);


    Vector tau =
        computeTau(input);


    Vector derivative =
        computeDerivative(
            target,
            tau
        );


    hiddenState =
        hiddenState
        + derivative * dt;


    return hiddenState;
}

void LTCLayer::resetState() {

    hiddenState.fill(0.0);
}

const Vector&
LTCLayer::getHiddenState() const {

    return hiddenState;
}

void LTCLayer::setInputWeight(
    std::size_t neuron,
    std::size_t input,
    double value
) {

    inputWeights(
        neuron,
        input
    ) = value;
}

void LTCLayer::setRecurrentWeight(
    std::size_t neuron,
    std::size_t previousNeuron,
    double value
) {

    recurrentWeights(
        neuron,
        previousNeuron
    ) = value;
}

void LTCLayer::setBias(
    std::size_t neuron,
    double value
) {

    bias[neuron] = value;
}

void LTCLayer::setTauInputWeight(
    std::size_t neuron,
    std::size_t input,
    double value
) {

    tauInputWeights(
        neuron,
        input
    ) = value;
}

void LTCLayer::setTauRecurrentWeight(
    std::size_t neuron,
    std::size_t previousNeuron,
    double value
) {

    tauRecurrentWeights(
        neuron,
        previousNeuron
    ) = value;
}

void LTCLayer::setTauBias(
    std::size_t neuron,
    double value
) {

    tauBias[neuron] =
        value;
}