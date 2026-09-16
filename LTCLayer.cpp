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

    for (const Vector& gradient : outputGradients) {
        if (gradient.size() != hiddenSize) {
            throw std::invalid_argument(
                "Invalid LTC output gradient size"
            );
        }
    }

    Matrix tauInputWeightGradients(
        hiddenSize,
        inputSize
    );

    Matrix tauRecurrentWeightGradients(
        hiddenSize,
        hiddenSize
    );

    Vector tauBiasGradients(
        hiddenSize
    );

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

        Vector tauGradient(
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

            const double difference =
                cache.target[i]
                - cache.previousState[i];

            tauGradient[i] =
                -stateGradient[i] * dt * difference
                / (cache.tau[i] * cache.tau[i]);
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

        Vector tauCombinedGradient(hiddenSize);

        for (std::size_t i = 0; i < hiddenSize; ++i) {
            const double sigmoidValue = cache.tau[i] - 1.0;
            const double sigmoidDerivative =
                sigmoidValue * (1.0 - sigmoidValue);

            tauCombinedGradient[i] =
                tauGradient[i] * sigmoidDerivative;
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

                tauInputWeightGradients(neuron, input) +=
                    tauCombinedGradient[neuron]
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

                tauRecurrentWeightGradients(neuron, previousNeuron) +=
                    tauCombinedGradient[neuron]
                    * cache.previousState[previousNeuron];
            }


            biasGradients[neuron] +=
                combinedGradient[neuron];

            tauBiasGradients[neuron] +=
                tauCombinedGradient[neuron];
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

                gradient +=
                    tauInputWeights(neuron, input)
                    * tauCombinedGradient[neuron];
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

                previousStateGradient[previousNeuron] +=
                    tauRecurrentWeights(neuron, previousNeuron)
                    * tauCombinedGradient[neuron];
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
            const double gradient = clipGradient(
                inputWeightGradients(neuron, input), 1.0
            );
            inputWeights(neuron, input) -=
                learningRate * gradient;

            const double tauGradient = clipGradient(
                tauInputWeightGradients(neuron, input), 1.0
            );
            tauInputWeights(neuron, input) -=
                learningRate * tauGradient;
        }


        for (
            std::size_t previousNeuron = 0;
            previousNeuron < hiddenSize;
            ++previousNeuron
        ) {

            const double gradient = clipGradient(
                recurrentWeightGradients(neuron, previousNeuron), 1.0
            );
            recurrentWeights(neuron, previousNeuron) -=
                learningRate * gradient;

            const double tauGradient = clipGradient(
                tauRecurrentWeightGradients(neuron, previousNeuron), 1.0
            );
            tauRecurrentWeights(neuron, previousNeuron) -=
                learningRate * tauGradient;
        }

        bias[neuron] -= learningRate * clipGradient(
            biasGradients[neuron], 1.0
        );
        tauBias[neuron] -= learningRate * clipGradient(
            tauBiasGradients[neuron], 1.0
        );
    }


    return inputGradients;
}

double LTCLayer::clipGradient(
    double gradient,
    double limit
) const {

    if (gradient > limit) {
        return limit;
    }

    if (gradient < -limit) {
        return -limit;
    }

    return gradient;
}
