#pragma once

#include "Vector.hpp"

namespace Activation {

    double tanh(double x);

    double sigmoid(double x);

    double tanhDerivative(double x);

    double sigmoidDerivative(double x);

    Vector tanh(const Vector& input);

    Vector sigmoid(const Vector& input);
}