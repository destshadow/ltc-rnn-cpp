#include "Matrix.hpp"

#include <stdexcept>


Matrix::Matrix()
    : rows(0),
      cols(0) {
}


Matrix::Matrix(
    std::size_t rows,
    std::size_t cols
)
    : rows(rows),
      cols(cols),
      data(rows * cols, 0.0) {
}


Matrix::Matrix(
    std::size_t rows,
    std::size_t cols,
    double initialValue
)
    : rows(rows),
      cols(cols),
      data(rows * cols, initialValue) {
}

std::size_t Matrix::getRows() const {
    return rows;
}

std::size_t Matrix::getCols() const {
    return cols;
}

double& Matrix::operator()(
    std::size_t row,
    std::size_t col
) {

    return data[row * cols + col];
}

const double& Matrix::operator()(
    std::size_t row,
    std::size_t col
) const {

    return data[row * cols + col];
}

Vector Matrix::operator*(
    const Vector& vector
) const {

    if (cols != vector.size()) {
        throw std::invalid_argument(
            "Matrix/vector size mismatch"
        );
    }

    Vector result(rows);

    for (std::size_t row = 0; row < rows; ++row) { //visita ogni riga, PER OGNI neurone

        double sum = 0.0;

        for (std::size_t col = 0; col < cols; ++col) { //visita tutti i pesi di quel neurone,  PER OGNI input

            sum += (*this)(row, col) * vector[col];
        }

        result[row] = sum; //output_neurone = somma
    }

    return result;
}