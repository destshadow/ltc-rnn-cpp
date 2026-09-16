#include <iostream>
#include <random>
#include <vector>

#include "../LTCNetwork.hpp"
#include "../SGDOptimizer.hpp"

double randomNoise(
    double min,
    double max
) {
    static std::mt19937 generator(12345);

    std::uniform_real_distribution<double>
        distribution(min, max);

    return distribution(generator);
}

std::vector<Vector>
makeIncreasingSequence() {

    const double base[] = {
        0.1,
        0.3,
        0.5,
        0.7
    };

    std::vector<Vector> sequence;

    for (double value : base) {

        Vector input(1);

        input[0] =
            value
            + randomNoise(
                -0.05,
                0.05
            );

        sequence.push_back(input);
    }

    return sequence;
}

std::vector<Vector>
makeDecreasingSequence() {

    const double base[] = {
        0.7,
        0.5,
        0.3,
        0.1
    };

    std::vector<Vector> sequence;

    for (double value : base) {

        Vector input(1);

        input[0] =
            value
            + randomNoise(
                -0.05,
                0.05
            );

        sequence.push_back(input);
    }

    return sequence;
}

int main() {

LTCNetwork network;

network.addLayer(
    1,
    6,
    0.5
);

network.setOutputLayer(1);

SGDOptimizer optimizer(0.01);

Vector targetIncreasing(1);
targetIncreasing[0] = 1.0;

Vector targetDecreasing(1);
targetDecreasing[0] = -1.0;

for (
    int epoch = 0;
    epoch < 3000;
    ++epoch
) {

    std::vector<Vector> increasing =
        makeIncreasingSequence();

    std::vector<Vector> decreasing =
        makeDecreasingSequence();


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


    if (epoch % 300 == 0) {

        std::cout
            << "Epoch "
            << epoch
            << " | Avg loss: "
            << (
                lossIncreasing
                + lossDecreasing
            ) / 2.0
            << '\n';
    }
}

std::vector<Vector> testIncreasing =
    makeIncreasingSequence();

std::vector<Vector> testDecreasing =
    makeDecreasingSequence();

Vector predictionIncreasing =
    network.predictSequence(
        testIncreasing
    );

Vector predictionDecreasing =
    network.predictSequence(
        testDecreasing
    );


std::cout
    << "\nUnseen increasing: "
    << predictionIncreasing[0]
    << '\n';

std::cout
    << "Unseen decreasing: "
    << predictionDecreasing[0]
    << '\n';

int correct = 0;
int total = 200;

for (int i = 0; i < 100; ++i) {

    auto increasing =
        makeIncreasingSequence();

    Vector prediction =
        network.predictSequence(
            increasing
        );

    if (prediction[0] > 0.0) {
        ++correct;
    }


    auto decreasing =
        makeDecreasingSequence();

    prediction =
        network.predictSequence(
            decreasing
        );

    if (prediction[0] < 0.0) {
        ++correct;
    }
}

double accuracy =
    static_cast<double>(correct)
    / total;

std::cout
    << "Accuracy: "
    << accuracy * 100.0
    << "%\n";

return accuracy >= 0.95 ? 0 : 1;
}
