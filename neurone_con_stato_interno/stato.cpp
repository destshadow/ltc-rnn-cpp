#include <iostream>

class Neuron {
private:
    double state;

public:
    Neuron() {
        state = 0.0;
    }

    void update(double input) {
        state = input;
    }

    double getState() const {
        return state;
    }
};

int main() {

    Neuron neuron;

    neuron.update(5.0);

    std::cout << neuron.getState() << std::endl;

    return 0;
}

/*
Concetto di Stato:
Immagina che il nostro neurone abbia una piccola memoria.
questo neurone non conserva realmente il passato.

*/