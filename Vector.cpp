#include "Vector.hpp"

#include <stdexcept>

Vector::Vector() {
}

Vector::Vector(std::size_t size)
    : data(size, 0.0) {
}

Vector::Vector(std::size_t size, double initialValue)
    : data(size, initialValue) {
}

std::size_t Vector::size() const {
    return data.size();
}

double& Vector::operator[](std::size_t index) {
    return data[index];
}

const double& Vector::operator[](std::size_t index) const {
    return data[index];
}

void Vector::fill(double value) {
    for (double& element : data) {
        element = value;
    }
}

Vector Vector::operator+(const Vector& other) const {

    if (size() != other.size()) {
        throw std::invalid_argument("Vector size mismatch");
    }

    Vector result(size());

    for (std::size_t i = 0; i < size(); ++i) {
        result[i] = data[i] + other[i];
    }

    return result;
}

Vector Vector::operator-(const Vector& other) const {

    if (size() != other.size()) {
        throw std::invalid_argument("Vector size mismatch");
    }

    Vector result(size());

    for (std::size_t i = 0; i < size(); ++i) {
        result[i] = data[i] - other[i];
    }

    return result;
}

Vector Vector::operator*(double scalar) const {

    Vector result(size());

    for (std::size_t i = 0; i < size(); ++i) {
        result[i] = data[i] * scalar;
    }

    return result;
}

double Vector::dot(const Vector& other) const {

    if (size() != other.size()) {
        throw std::invalid_argument("Vector size mismatch");
    }

    double result = 0.0;

    for (std::size_t i = 0; i < size(); ++i) {
        result += data[i] * other[i];
    }

    return result;
}