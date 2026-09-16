#include "LTCNeuron.hpp"

#include <cmath>


namespace {

double sigmoid(double x) {

    return 1.0 / (1.0 + std::exp(-x));
}

}


LTCNeuron::LTCNeuron(
    double inputWeight,
    double recurrentWeight,
    double bias,

    double tauInputWeight,
    double tauRecurrentWeight,
    double tauBias,

    double dt
)
    : state(0.0),
      inputWeight(inputWeight),
      recurrentWeight(recurrentWeight),
      bias(bias),

      tauInputWeight(tauInputWeight),
      tauRecurrentWeight(tauRecurrentWeight),
      tauBias(tauBias),

      dt(dt) {
}


double LTCNeuron::computeTarget(
    double input
) const {

    double weightedInput =
        inputWeight * input;

    double recurrentContribution =
        recurrentWeight * state;

    double combined =
        weightedInput
        + recurrentContribution
        + bias;

    return std::tanh(combined);
}


double LTCNeuron::computeTau(
    double input
) const {

    double value =
        tauInputWeight * input
        + tauRecurrentWeight * state
        + tauBias;

    return 1.0 + sigmoid(value);
}


double LTCNeuron::computeDerivative(
    double target,
    double tau
) const {

    return (target - state) / tau;
}


double LTCNeuron::forward(
    double input
) {

    double target =
        computeTarget(input);

    double tau =
        computeTau(input);

    double derivative =
        computeDerivative(
            target,
            tau
        );

    state =
        state + dt * derivative;

    return state;
}


double LTCNeuron::getState() const {

    return state;
}


void LTCNeuron::resetState() {

    state = 0.0;
}