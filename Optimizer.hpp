#pragma once

class Optimizer {

public:

    virtual ~Optimizer() = default;

    virtual double update(
        double parameter,
        double gradient
    ) = 0;
};