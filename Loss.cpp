#include "Loss.hpp"

#include <stdexcept>


double Loss::meanSquaredError(
    const Vector& prediction,
    const Vector& target
) {

    if (prediction.size() != target.size()) {

        throw std::invalid_argument(
            "Prediction and target size mismatch"
        );
    }

    if (prediction.size() == 0) {

        throw std::invalid_argument(
            "Cannot compute loss on empty vectors"
        );
    }


    double sum = 0.0;


    for (
        std::size_t i = 0;
        i < prediction.size();
        ++i
    ) {

        double error =
            prediction[i] - target[i];

        sum +=
            error * error;
    }


    return sum / prediction.size();
}

Vector Loss::meanSquaredErrorGradient(
    const Vector& prediction,
    const Vector& target
) {
    if (prediction.size() != target.size()) {
        throw std::invalid_argument(
            "Prediction and target size mismatch"
        );
    }

    if (prediction.size() == 0) {
        throw std::invalid_argument(
            "Cannot compute loss gradient on empty vectors"
        );
    }

    Vector gradient(prediction.size());
    const double scale = 2.0 / prediction.size();

    for (std::size_t i = 0; i < prediction.size(); ++i) {
        gradient[i] = scale * (prediction[i] - target[i]);
    }

    return gradient;
}
