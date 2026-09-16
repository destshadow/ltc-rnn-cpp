#include <iostream>
#include <vector>

#include "LTCNetwork.hpp"
#include "SGDOptimizer.hpp"

int main() {

    LTCNetwork network;

    network.addLayer(
        1,      // input size
        4,      // hidden neurons
        0.05    // dt
    );

    network.setOutputLayer(1);


    SGDOptimizer optimizer(0.01);


    std::vector<Vector> sequence;


    for (double value : {0.1, 0.3, 0.5, 0.7}) {

        Vector input(1);

        input[0] = value;

        sequence.push_back(input);
    }


    Vector target(1);

    target[0] = 1.0;


    for (int epoch = 0; epoch < 1000; ++epoch) {

        double loss =
            network.trainSequence(
                sequence,
                target,
                optimizer
            );


        if (epoch % 100 == 0) {

            Vector prediction =
                network.predictSequence(
                    sequence
                );

            std::cout
                << "Epoch: "
                << epoch
                << " | Prediction: "
                << prediction[0]
                << " | Loss: "
                << loss
                << '\n';
        }
    }


    Vector finalPrediction =
        network.predictSequence(
            sequence
        );


    std::cout
        << "\nFinal prediction: "
        << finalPrediction[0]
        << '\n';

    std::cout
        << "Target: "
        << target[0]
        << '\n';


    return 0;
}

/*
g++ -std=c++17 *.cpp -o main
*/
