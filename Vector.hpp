#pragma once

#include <vector>
#include <cstddef>

class Vector {
private:
    std::vector<double> data;

public:
    Vector();
    Vector(std::size_t size);
    Vector(std::size_t size, double initialValue);

    std::size_t size() const;

    double& operator[](std::size_t index);
    const double& operator[](std::size_t index) const;

    void fill(double value);

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(double scalar) const;

    double dot(const Vector& other) const;
};