#pragma once

#include <vector>
#include <cstddef>

#include "Vector.hpp"

class Matrix {

private:

    std::size_t rows;
    std::size_t cols;

    std::vector<double> data;

public:

    Matrix();

    Matrix(std::size_t rows, std::size_t cols);

    Matrix(
        std::size_t rows,
        std::size_t cols,
        double initialValue
    );

    std::size_t getRows() const;
    std::size_t getCols() const;

    double& operator()(
        std::size_t row,
        std::size_t col
    );

    const double& operator()(
        std::size_t row,
        std::size_t col
    ) const;

    Vector operator*(const Vector& vector) const;
};