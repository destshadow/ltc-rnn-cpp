#include <iostream>

#include "LTCLayer.hpp"

int main() {

    LTCLayer layer(
        1,
        3,
        0.05
    );


    Vector input(1);


    for (
        int step = 0;
        step < 10;
        ++step
    ) {

        input[0] =
            static_cast<double>(step)
            / 10.0;

        layer.forward(input);


        std::cout
            << "Step: "
            << step
            << " | history: "
            << layer.getHistorySize()
            << '\n';
    }


    return 0;
}

/*
g++ -std=c++17 *.cpp -o main
*/