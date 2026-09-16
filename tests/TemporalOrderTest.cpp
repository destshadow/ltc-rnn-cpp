#include <iostream>
#include <vector>
#include <cmath>

#include "../LTCNetwork.hpp"
#include "../SGDOptimizer.hpp"


int main() {

    LTCNetwork network;

    network.addLayer(
        1,
        6,
        0.5
    );

    network.setOutputLayer(1);

    SGDOptimizer optimizer(0.01);


    std::vector<Vector> increasing;
    std::vector<Vector> decreasing;


    for (double value : {0.1, 0.3, 0.5, 0.7}) {

        Vector input(1);
        input[0] = value;

        increasing.push_back(input);
    }


    for (double value : {0.7, 0.5, 0.3, 0.1}) {

        Vector input(1);
        input[0] = value;

        decreasing.push_back(input);
    }


    Vector targetIncreasing(1);
    targetIncreasing[0] = 1.0;

    Vector targetDecreasing(1);
    targetDecreasing[0] = -1.0;


    for (int epoch = 0; epoch < 2000; ++epoch) {

        network.trainSequence(
            increasing,
            targetIncreasing,
            optimizer
        );

        network.trainSequence(
            decreasing,
            targetDecreasing,
            optimizer
        );
    }


    Vector predictionIncreasing =
        network.predictSequence(
            increasing
        );

    Vector predictionDecreasing =
        network.predictSequence(
            decreasing
        );


    std::cout
        << "Increasing: "
        << predictionIncreasing[0]
        << '\n';

    std::cout
        << "Decreasing: "
        << predictionDecreasing[0]
        << '\n';


    bool increasingOk =
        predictionIncreasing[0] > 0.8;

    bool decreasingOk =
        predictionDecreasing[0] < -0.8;


    if (
        increasingOk
        && decreasingOk
    ) {

        std::cout
            << "TEST PASSED\n";

        return 0;
    }


    std::cout
        << "TEST FAILED\n";

    return 1;
}
