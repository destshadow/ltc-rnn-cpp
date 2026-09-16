#pragma once

#include "Optimizer.hpp"

class SGDOptimizer : public Optimizer {

private:

    double learningRate;

public:

    explicit SGDOptimizer(
        double learningRate
    );

    double update(
        double parameter,
        double gradient
    ) override;

    double getLearningRate() const;
};