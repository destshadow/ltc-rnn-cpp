#pragma once

class LTCNeuron {

private:

    double state; /*Importantissimo */

    double inputWeight;
    double recurrentWeight;
    double bias;

    double tauInputWeight;
    double tauRecurrentWeight;
    double tauBias;

    double dt;

    double computeTau(double input) const;

public:

    LTCNeuron(
        double inputWeight,
        double recurrentWeight,
        double bias,

        double tauInputWeight,
        double tauRecurrentWeight,
        double tauBias,

        double dt
    );

    double forward(double input);

    double getState() const;

    void resetState();
};