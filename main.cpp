#include <iostream>
#include <vector>

#include "LTCNetwork.hpp"


int main() {

    LTCNetwork network;


    network.addLayer(
        1,
        6,
        0.05
    );

    network.setOutputLayer(
        1
    );


    std::vector<Vector> increasing;
    std::vector<Vector> decreasing;


    for (double value :
         {0.1, 0.3, 0.5, 0.7}) {

        Vector x(1);

        x[0] = value;

        increasing.push_back(x);
    }


    for (double value :
         {0.7, 0.5, 0.3, 0.1}) {

        Vector x(1);

        x[0] = value;

        decreasing.push_back(x);
    }


    Vector targetIncreasing(1);
    targetIncreasing[0] = 1.0;


    Vector targetDecreasing(1);
    targetDecreasing[0] = -1.0;


    double learningRate =
        0.01;


    for (
        int epoch = 0;
        epoch < 1000;
        ++epoch
    ) {

        double loss1 =
            network.trainSequence(
                increasing,
                targetIncreasing,
                learningRate
            );


        double loss2 =
            network.trainSequence(
                decreasing,
                targetDecreasing,
                learningRate
            );


        if (epoch % 100 == 0) {

            std::cout
                << "Epoch "
                << epoch
                << " | Loss = "
                << (loss1 + loss2) / 2.0
                << '\n';
        }
    }

    Vector prediction1 = network.predictSequence(increasing);
    Vector prediction2 = network.predictSequence(decreasing);

    std::cout << "\nIncreasing: " << prediction1[0] << '\n';
    std::cout << "Decreasing: " << prediction2[0] << '\n';


    return 0;
}

/*
g++ -std=c++17 *.cpp -o main
*/
