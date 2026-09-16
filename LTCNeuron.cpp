#include "LTCNeuron.hpp"

#include <cmath>


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

double LTCNeuron::forward(double input) {

    double weightedInput =
        inputWeight * input;

    double recurrentContribution =
        recurrentWeight * state;

    double combined =
        weightedInput
        + recurrentContribution
        + bias;

    double target =
        std::tanh(combined);

    double tau =
        computeTau(input);

    double derivative =
        (target - state) / tau;

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

double sigmoid(double x) {

    return 1.0 / (1.0 + std::exp(-x));
}

double LTCNeuron::computeTau(double input) const {

    double value =
        tauInputWeight * input
        + tauRecurrentWeight * state
        + tauBias;

    return 1.0 + sigmoid(value);
}