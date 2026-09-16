#include "Activation.hpp"

#include <cmath>


double Activation::tanh(double x) {

    return std::tanh(x);
}

Vector Activation::tanh(const Vector& input) {

    Vector result(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {

        result[i] = std::tanh(input[i]);
    }

    return result;
}

double Activation::sigmoid(double x) {

    return 1.0 / (1.0 + std::exp(-x));
}

Vector Activation::sigmoid(const Vector& input) {

    Vector result(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {

        result[i] = sigmoid(input[i]);
    }

    return result;
}

double Activation::tanhDerivative(double x) {

    double t = std::tanh(x);

    return 1.0 - t * t;
}

double Activation::sigmoidDerivative(double x) {

    double s = sigmoid(x);

    return s * (1.0 - s);
}