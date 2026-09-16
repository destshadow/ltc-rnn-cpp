#pragma once

#include "Vector.hpp"

namespace Loss {

    double meanSquaredError(
        const Vector& prediction,
        const Vector& target
    );

    Vector meanSquaredErrorGradient(
        const Vector& prediction,
        const Vector& target
    );
}