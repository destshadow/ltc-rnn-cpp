#include <iostream>

#include "LTCLayer.hpp"


int main() {

    LTCLayer layer(
        2,
        4,
        0.1
    );


    Vector input(2);

    input[0] = 1.0;
    input[1] = 0.5;


    for (
        int step = 0;
        step < 20;
        ++step
    ) {

        Vector output =
            layer.forward(input);


        std::cout
            << "Step "
            << step
            << ": ";

        for (
            std::size_t i = 0;
            i < output.size();
            ++i
        ) {

            std::cout
                << output[i]
                << " ";
        }

        std::cout << '\n';
    }
}

/*
g++ -std=c++17 *.cpp -o main
*/