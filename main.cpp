#include <iostream>

#include "DenseLayer.hpp"

int main() {

    DenseLayer layer(3, 1);

    layer.setWeight(0, 0, 0.5);
    layer.setWeight(0, 1, 0.2);
    layer.setWeight(0, 2, -0.1);

    layer.setBias(0, 0.1);


    Vector input(3);

    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;


    Vector output =
        layer.forward(input);


    std::cout
        << "Output: "
        << output[0]
        << '\n';

    return 0;
}

//output corretto: Output: 0.7

/*
g++ -std=c++17 *.cpp -o main
*/