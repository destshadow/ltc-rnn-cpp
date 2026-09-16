#include <iostream>

#include "LTCLayer.hpp"


int main() {

    LTCLayer layer(
        1,
        2,
        0.1
    );


    layer.setInputWeight(
        0, 0, 1.0
    );

    layer.setInputWeight(
        1, 0, -0.7
    );


    layer.setRecurrentWeight(
        0, 0, 0.5
    );

    layer.setRecurrentWeight(
        0, 1, 0.2
    );

    layer.setRecurrentWeight(
        1, 0, -0.1
    );

    layer.setRecurrentWeight(
        1, 1, 0.6
    );


    layer.setTauInputWeight(
        0, 0, 1.0
    );

    layer.setTauInputWeight(
        1, 0, -1.0
    );


    Vector input(1);


    for (
        int step = 0;
        step < 20;
        ++step
    ) {

        if (step < 5) {
            input[0] = 1.0;
        }
        else {
            input[0] = 0.0;
        }


        Vector output =
            layer.forward(input);


        std::cout
            << "Step "
            << step
            << " | input = "
            << input[0]
            << " | h0 = "
            << output[0]
            << " | h1 = "
            << output[1]
            << '\n';
    }
}

/*
g++ -std=c++17 *.cpp -o main
*/