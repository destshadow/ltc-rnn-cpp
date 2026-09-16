#include <iostream>
#include <vector>

#include "LTCNetwork.hpp"
#include "SGDOptimizer.hpp"

//test 2 su due sequences, una crescente e una decrescente, con target rispettivamente 1 e -1

int main() {

    LTCNetwork network;

    network.addLayer(
        1,      // input size
        6,      // hidden neurons
        0.5    // dt /* prima era 0.05 e non andava bene ora invece è perfetto 0.5 è il numero giusto */
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

        double lossIncreasing =
            network.trainSequence(
                increasing,
                targetIncreasing,
                optimizer
            );

        double lossDecreasing =
            network.trainSequence(
                decreasing,
                targetDecreasing,
                optimizer
            );


        if (epoch % 200 == 0) {

            Vector predictionIncreasing =
                network.predictSequence(
                    increasing
                );

            Vector predictionDecreasing =
                network.predictSequence(
                    decreasing
                );


            std::cout
                << "Epoch "
                << epoch
                << '\n';

            std::cout
                << "  Increasing: "
                << predictionIncreasing[0]
                << " | target: 1"
                << '\n';

            std::cout
                << "  Decreasing: "
                << predictionDecreasing[0]
                << " | target: -1"
                << '\n';

            std::cout
                << "  Avg loss: "
                << (
                    lossIncreasing
                    + lossDecreasing
                ) / 2.0
                << "\n\n";
        }
    }


    return 0;
}

/*
g++ -std=c++17 *.cpp -o main
*/
