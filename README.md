# LTC RNN in C++

Un piccolo progetto didattico in C++17 per esplorare neuroni con stato, reti ricorrenti e una semplice dinamica ispirata ai *Liquid Time-Constant networks* (LTC).

Il programma attuale mostra un esempio di `DenseLayer` con tre input e un output. Il progetto contiene anche un neurone `LTCNeuron`, un livello `LTCLayer`, una `LTCNetwork` che combina più livelli LTC, una classe `RNNLayer` e le operazioni di base su vettori e matrici.

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

`main.cpp` crea una `DenseLayer` con tre input e un output. Imposta i pesi a `0.5`, `0.2` e `-0.1`, il bias a `0.1` e l'input a `[1, 2, 3]`. Il risultato è `Output: 0.7`.

La `DenseLayer` calcola `weights * input + bias`. `LTCLayer` usa invece uno stato ricorrente e una costante di tempo variabile per ciascun neurone; `LTCNetwork` permette di concatenare più livelli LTC con dimensioni compatibili.

## Struttura

| File | Contenuto |
| --- | --- |
| `main.cpp` | Esempio eseguibile del livello denso |
| `DenseLayer.hpp/.cpp` | Livello denso con pesi, bias e prodotto matrice-vettore |
| `LTCNetwork.hpp/.cpp` | Sequenza di livelli LTC |
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
