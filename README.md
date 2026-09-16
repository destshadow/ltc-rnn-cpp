# LTC RNN in C++

Un piccolo progetto didattico in C++17 per esplorare neuroni con stato, reti ricorrenti e una semplice dinamica ispirata ai *Liquid Time-Constant networks* (LTC).

Il programma attuale mostra un semplice addestramento di una `DenseLayer` con due input e un output. Il progetto contiene anche un neurone `LTCNeuron`, un livello `LTCLayer`, una `LTCNetwork` che combina più livelli LTC, una classe `RNNLayer` e le operazioni di base su vettori e matrici.

> Progetto sperimentale: l'esempio addestra solo una `DenseLayer`; non include ancora l'addestramento dei livelli LTC o la gestione di dataset.

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

`main.cpp` crea una `DenseLayer` con due input e un output. Imposta entrambi i pesi a `0.5`, il bias a `0`, l'input a `[1, 1]` e l'obiettivo a `2`. Per venti epoche calcola la predizione, la perdita quadratica media e il suo gradiente, poi aggiorna pesi e bias con `backward`.

La predizione parte da `1` e si avvicina a `2`, mentre la perdita diminuisce. `LTCNetwork` permette di concatenare più livelli LTC con dimensioni compatibili e di aggiungere un livello denso di uscita opzionale.

## Struttura

| File | Contenuto |
| --- | --- |
| `main.cpp` | Esempio di addestramento del livello denso |
| `DenseLayer.hpp/.cpp` | Livello denso con forward e aggiornamento dei parametri |
| `Loss.hpp/.cpp` | Perdita quadratica media e relativo gradiente |
| `LTCNetwork.hpp/.cpp` | Sequenza di livelli LTC con uscita densa opzionale |
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

Il codice è pensato per apprendere e sperimentare. `RNNLayer` implementa un aggiornamento ricorrente con `tanh`; non usa la dinamica a costante di tempo variabile di `LTCNeuron`. Non ci sono ancora API per addestrare i pesi dei livelli LTC né controlli completi sugli indici di vettori e matrici.
