# LTC RNN in C++

Un piccolo progetto didattico in C++17 per esplorare neuroni con stato, reti ricorrenti e una semplice dinamica ispirata ai *Liquid Time-Constant networks* (LTC).

Il programma attuale mostra l'addestramento del livello denso di uscita di una `LTCNetwork` con due input, quattro neuroni LTC e un output. Il progetto contiene anche un neurone `LTCNeuron`, un livello `LTCLayer`, una classe `RNNLayer` e le operazioni di base su vettori e matrici.

> Progetto sperimentale: `trainStep` aggiorna solo il livello denso di uscita; non addestra i livelli LTC né gestisce dataset.

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

`main.cpp` crea una `LTCNetwork` con un livello LTC da quattro neuroni e un livello denso di uscita. Usa l'input `[1, 0.5]` e l'obiettivo `0.8`. Per cento epoche azzera lo stato ricorrente, calcola la perdita quadratica media e aggiorna i pesi e il bias del livello denso con `trainStep`.

La perdita diminuisce e la predizione finale si avvicina all'obiettivo. `LTCNetwork` permette anche di concatenare più livelli LTC con dimensioni compatibili.

Ogni chiamata a `LTCLayer::forward` registra i dati del passo nella cronologia del livello. `clearHistory()` la svuota; `LTCNetwork::resetSequence()` azzera sia gli stati ricorrenti sia le cronologie dei livelli. La cronologia non viene ancora usata per aggiornare i pesi LTC.

## Struttura

| File | Contenuto |
| --- | --- |
| `main.cpp` | Esempio di addestramento dell'uscita di `LTCNetwork` |
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
