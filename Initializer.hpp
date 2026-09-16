#pragma once

#include <cstddef>

#include "Vector.hpp"
#include "Matrix.hpp"

namespace Initializer {

    double randomValue(
        double min,
        double max
    );

    void randomize(
        Vector& vector,
        double min,
        double max
    );

    void randomize(
        Matrix& matrix,
        double min,
        double max
    );
}