#include <iostream>

class RNNNeuron {
private:
    double state;

public:
    RNNNeuron() {
        state = 0.0;
    }

    void update(double input) {

        state = 0.5 * state + 0.5 * input;
    }

    double getState() const {
        return state;
    }
};

int main() {

    RNNNeuron neuron;

    neuron.update(10.0);
    neuron.update(0.0);
    neuron.update(0.0);
    neuron.update(0.0);


    std::cout << neuron.getState() << std::endl;

    return 0;
}