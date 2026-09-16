#include <iostream>

#include "LTCNeuron.hpp"

int main() {

    LTCNeuron neuron(
        1.0,    // inputWeight
        0.5,    // recurrentWeight
        0.0,    // bias

        2.0,    // tauInputWeight
        1.0,    // tauRecurrentWeight
        0.0,    // tauBias

        0.1     // dt
    );


    for (int step = 0; step < 20; ++step) {

        double input;

        if (step < 5) {
            input = 1.0;
        }
        else {
            input = 0.0;
        }


        double output =
            neuron.forward(input);


        std::cout
            << "Step "
            << step
            << " | input = "
            << input
            << " | state = "
            << output
            << '\n';
    }

    return 0;
}

/*
g++ -std=c++17 *.cpp -o main
*/