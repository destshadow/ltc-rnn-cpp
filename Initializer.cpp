#include "Initializer.hpp"

#include <random>


namespace {

std::mt19937& getGenerator() {

    static std::random_device rd;
    static std::mt19937 generator(rd());

    return generator;
}

}


double Initializer::randomValue(
    double min,
    double max
) {

    std::uniform_real_distribution<double>
        distribution(min, max);

    return distribution(
        getGenerator()
    );
}

void Initializer::randomize(
    Vector& vector,
    double min,
    double max
) {

    for (
        std::size_t i = 0;
        i < vector.size();
        ++i
    ) {

        vector[i] =
            randomValue(
                min,
                max
            );
    }
}

void Initializer::randomize(
    Matrix& matrix,
    double min,
    double max
) {

    for (
        std::size_t row = 0;
        row < matrix.getRows();
        ++row
    ) {

        for (
            std::size_t col = 0;
            col < matrix.getCols();
            ++col
        ) {

            matrix(row, col) =
                randomValue(
                    min,
                    max
                );
        }
    }
}