#include <iostream>

#include "DenseLayer.hpp"
#include "Loss.hpp"

int main() {

    DenseLayer layer(
        2,
        1
    );

    layer.setWeight(
        0,
        0,
        0.5
    );

    layer.setWeight(
        0,
        1,
        0.5
    );

    layer.setBias(
        0,
        0.0
    );


    Vector input(2);

    input[0] = 1.0;
    input[1] = 1.0;


    Vector target(1);

    target[0] = 2.0;


    double learningRate =
        0.1;


    for (
        int epoch = 0;
        epoch < 20;
        ++epoch
    ) {

        Vector prediction =
            layer.forward(input);


        double loss =
            Loss::meanSquaredError(
                prediction,
                target
            );


        Vector lossGradient =
            Loss::meanSquaredErrorGradient(
                prediction,
                target
            );


        layer.backward(
            lossGradient,
            learningRate
        );


        std::cout
            << "Epoch "
            << epoch
            << " | prediction = "
            << prediction[0]
            << " | loss = "
            << loss
            << '\n';
    }

    return 0;
}

/*
g++ -std=c++17 *.cpp -o main
*/