# LTC RNN in C++

Un piccolo progetto didattico in C++17 per esplorare neuroni con stato, reti ricorrenti e una semplice dinamica ispirata ai *Liquid Time-Constant networks* (LTC).

Il programma attuale simula una `LTCLayer` con due neuroni: riceve un impulso per i primi cinque passi e mostra come evolvono i due stati nei passi successivi. Il progetto contiene anche un neurone `LTCNeuron` singolo, una classe `RNNLayer` e le operazioni di base su vettori e matrici.

> Progetto sperimentale: il codice non include addestramento o gestione di dataset.

## Compilazione ed esecuzione

Serve un compilatore C++ con supporto a C++17, per esempio `g++`.

```bash
g++ -std=c++17 -Wall -Wextra -Wpedantic *.cpp -o main
./main
```

Esegui i comandi dalla cartella del repository. In alternativa:

```bash
make
make run
```

`make clean` rimuove l'eseguibile generato.

## Come funziona l'esempio

`main.cpp` crea una `LTCLayer` con un input e due neuroni, entrambi con stato iniziale pari a zero. A ogni passo `forward(input)`:

1. combina l'input corrente con lo stato precedente e applica `tanh` per ottenere un valore obiettivo;
2. calcola per ciascun neurone una costante di tempo variabile `tau = 1 + sigmoid(W_tau_x * input + W_tau_h * state + b_tau)`;
3. aggiorna ogni stato con `state += dt * (target - state) / tau`.

Con i parametri presenti in `main.cpp`, l'input vale `1` nei primi cinque passi e `0` nei successivi quindici. L'output stampa passo, input e stati `h0` e `h1`.

## Struttura

| File | Contenuto |
| --- | --- |
| `main.cpp` | Esempio eseguibile del livello LTC |
| `LTCLayer.hpp/.cpp` | Livello ricorrente con costanti di tempo variabili |
| `LTCNeuron.hpp/.cpp` | Neurone con stato e costante di tempo variabile |
| `RNNLayer.hpp/.cpp` | Livello ricorrente con pesi, bias e stato nascosto |
| `Vector.hpp/.cpp` | Vettore di `double` e operazioni aritmetiche |
| `Matrix.hpp/.cpp` | Matrice di `double` e prodotto matrice-vettore |
| `Activation.hpp/.cpp` | Funzioni di attivazione e derivate |
| `Initializer.hpp/.cpp` | Inizializzazione casuale di vettori e matrici |
| `neurone_con_stato_interno/` | Due esempi autonomi per capire lo stato di un neurone |
| `references/` | I due PDF di riferimento presenti nella cartella originale del progetto |

Gli esempi in `neurone_con_stato_interno/` hanno ciascuno una propria funzione `main` e si compilano separatamente:

```bash
g++ -std=c++17 neurone_con_stato_interno/stato.cpp -o stato
g++ -std=c++17 neurone_con_stato_interno/rnn.cpp -o rnn
```

## Stato del progetto

Il codice è pensato per apprendere e sperimentare. `RNNLayer` implementa un aggiornamento ricorrente con `tanh`; non usa la dinamica a costante di tempo variabile di `LTCNeuron`. Non ci sono ancora API per addestrare i pesi né controlli completi sugli indici di vettori e matrici.
