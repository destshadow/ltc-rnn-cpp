#include "LTCLayer.hpp"

#include "Activation.hpp"
#include "Initializer.hpp"

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
    Initializer::randomize(inputWeights, -0.5, 0.5);
    Initializer::randomize(recurrentWeights, -0.5, 0.5);
    Initializer::randomize(tauInputWeights, -0.5, 0.5);
    Initializer::randomize(tauRecurrentWeights, -0.5, 0.5);
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

    if (input.size() != inputSize) {

        throw std::invalid_argument(
            "Invalid LTC input size"
        );
    }


    Vector previousState =
        hiddenState;


    Vector target =
        computeTarget(input);


    Vector tau =
        computeTau(input);


    Vector derivative =
        computeDerivative(
            target,
            tau
        );


    history.push_back({
        input,
        previousState,
        target,
        tau,
        derivative
    });


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

std::size_t LTCLayer::getInputSize() const {

    return inputSize;
}


std::size_t LTCLayer::getHiddenSize() const {

    return hiddenSize;
}

void LTCLayer::clearHistory() {

    history.clear();
}

std::size_t LTCLayer::getHistorySize() const {

    return history.size();
}

std::vector<Vector> LTCLayer::backward(
    const std::vector<Vector>& outputGradients,
    double learningRate
) {

    if (
        outputGradients.size()
        != history.size()
    ) {
        throw std::invalid_argument(
            "Gradient history size mismatch"
        );
    }

    if (history.empty()) {
        return {};
    }

    Matrix inputWeightGradients(
        hiddenSize,
        inputSize
    );

    Matrix recurrentWeightGradients(
        hiddenSize,
        hiddenSize
    );

    Vector biasGradients(
        hiddenSize
    );

    std::vector<Vector> inputGradients;

    inputGradients.reserve(
        history.size()
    );

    for (
        std::size_t t = 0;
        t < history.size();
        ++t
    ) {

        inputGradients.emplace_back(
            inputSize
        );
    }


    Vector futureStateGradient(
        hiddenSize
    );

    futureStateGradient.fill(
        0.0
    );


    for (
        std::size_t t = history.size();
        t-- > 0;
    ) {

        const LTCStepCache& cache =
            history[t];


        Vector stateGradient(
            hiddenSize
        );

        for (
            std::size_t i = 0;
            i < hiddenSize;
            ++i
        ) {

            stateGradient[i] =
                outputGradients[t][i]
                + futureStateGradient[i];
        }


        Vector targetGradient(
            hiddenSize
        );

        for (
            std::size_t i = 0;
            i < hiddenSize;
            ++i
        ) {

            targetGradient[i] =
                stateGradient[i]
                * dt
                / cache.tau[i];
        }


        Vector combinedGradient(
            hiddenSize
        );

        for (
            std::size_t i = 0;
            i < hiddenSize;
            ++i
        ) {

            double tanhGradient =
                1.0
                - cache.target[i]
                * cache.target[i];

            combinedGradient[i] =
                targetGradient[i]
                * tanhGradient;
        }

        for (
            std::size_t neuron = 0;
            neuron < hiddenSize;
            ++neuron
        ) {

            for (
                std::size_t input = 0;
                input < inputSize;
                ++input
            ) {

                inputWeightGradients(
                    neuron,
                    input
                ) +=
                    combinedGradient[neuron]
                    * cache.input[input];
            }


            for (
                std::size_t previousNeuron = 0;
                previousNeuron < hiddenSize;
                ++previousNeuron
            ) {

                recurrentWeightGradients(
                    neuron,
                    previousNeuron
                ) +=
                    combinedGradient[neuron]
                    * cache.previousState[
                        previousNeuron
                    ];
            }


            biasGradients[neuron] +=
                combinedGradient[neuron];
        }


        for (
            std::size_t input = 0;
            input < inputSize;
            ++input
        ) {

            double gradient = 0.0;

            for (
                std::size_t neuron = 0;
                neuron < hiddenSize;
                ++neuron
            ) {

                gradient +=
                    inputWeights(
                        neuron,
                        input
                    )
                    * combinedGradient[
                        neuron
                    ];
            }

            inputGradients[t][input] =
                gradient;
        }


        Vector previousStateGradient(
            hiddenSize
        );

        for (
            std::size_t neuron = 0;
            neuron < hiddenSize;
            ++neuron
        ) {

            previousStateGradient[neuron] =
                stateGradient[neuron]
                * (
                    1.0
                    - dt
                    / cache.tau[neuron]
                );
        }


        for (
            std::size_t previousNeuron = 0;
            previousNeuron < hiddenSize;
            ++previousNeuron
        ) {

            for (
                std::size_t neuron = 0;
                neuron < hiddenSize;
                ++neuron
            ) {

                previousStateGradient[
                    previousNeuron
                ] +=
                    recurrentWeights(
                        neuron,
                        previousNeuron
                    )
                    * combinedGradient[
                        neuron
                    ];
            }
        }


        futureStateGradient =
            previousStateGradient;
    }


    for (
        std::size_t neuron = 0;
        neuron < hiddenSize;
        ++neuron
    ) {

        for (
            std::size_t input = 0;
            input < inputSize;
            ++input
        ) {

            inputWeights(
                neuron,
                input
            ) -=
                learningRate
                * inputWeightGradients(
                    neuron,
                    input
                );
        }


        for (
            std::size_t previousNeuron = 0;
            previousNeuron < hiddenSize;
            ++previousNeuron
        ) {

            recurrentWeights(
                neuron,
                previousNeuron
            ) -=
                learningRate
                * recurrentWeightGradients(
                    neuron,
                    previousNeuron
                );
        }


        bias[neuron] -=
            learningRate
            * biasGradients[neuron];
    }


    return inputGradients;
}
