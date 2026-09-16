#include <iostream>

#include "LTCNetwork.hpp"


int main() {

    LTCNetwork network;


    network.addLayer(
        2,
        4,
        0.05
    );


    network.setOutputLayer(
        1
    );


    Vector input(2);

    input[0] = 1.0;
    input[1] = 0.5;


    Vector target(1);

    target[0] = 0.8;


    double learningRate =
        0.01;


    for (
        int epoch = 0;
        epoch < 100;
        ++epoch
    ) {

        network.resetState();

        double loss =
            network.trainStep(
                input,
                target,
                learningRate
            );


        if (epoch % 10 == 0) {

            std::cout
                << "Epoch "
                << epoch
                << " | Loss = "
                << loss
                << '\n';
        }
    }


    network.resetState();

    Vector prediction =
        network.forward(input);


    std::cout
        << "\nFinal prediction: "
        << prediction[0]
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