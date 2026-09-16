# LTC RNN in C++

Un piccolo progetto didattico in C++17 per esplorare neuroni con stato, reti ricorrenti e una semplice dinamica ispirata ai *Liquid Time-Constant networks* (LTC).

Il programma attuale addestra una `LTCNetwork` su una sequenza crescente e una decrescente. Il progetto contiene anche un neurone `LTCNeuron`, una classe `RNNLayer` e le operazioni di base su vettori e matrici.

> Progetto sperimentale: `trainSequence` propaga gradienti attraverso i passi e aggiorna i livelli LTC e l'uscita densa. L'esempio usa solo due sequenze di addestramento, quindi non misura la capacità di generalizzare.

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

Per compilare ed eseguire il test dell'ordine temporale:

```bash
make test
```

Il test allena la rete sulle due sequenze dell'esempio e controlla che le relative predizioni superino le soglie `0.8` e `-0.8`. I pesi iniziali sono casuali e il test non verifica sequenze mai viste durante l'addestramento.

## Come funziona l'esempio

`main.cpp` crea una `LTCNetwork` con un livello LTC da sei neuroni e un'uscita densa. Allena la rete sulla sequenza `[0.1, 0.3, 0.5, 0.7]` verso il target `1` e sulla sequenza inversa verso `-1`, mostrando periodicamente le predizioni e la perdita media.

`LTCNetwork` permette di concatenare più livelli LTC con dimensioni compatibili e di aggiungere un livello denso di uscita opzionale. Il suo `trainStep` aggiorna solo il livello denso di uscita.

Ogni chiamata a `LTCLayer::forward` registra i dati del passo nella cronologia del livello. `clearHistory()` la svuota; `LTCNetwork::resetSequence()` azzera sia gli stati ricorrenti sia le cronologie. `trainSequence()` usa la cronologia per aggiornare i pesi LTC.

## Struttura

| File | Contenuto |
| --- | --- |
| `main.cpp` | Esempio di addestramento su due sequenze |
| `DenseLayer.hpp/.cpp` | Livello denso con forward e aggiornamento dei parametri |
| `Loss.hpp/.cpp` | Perdita quadratica media e relativo gradiente |
| `Optimizer.hpp`, `SGDOptimizer.hpp/.cpp` | Interfaccia di aggiornamento dei parametri e discesa del gradiente |
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

Il codice è pensato per apprendere e sperimentare. `RNNLayer` implementa un aggiornamento ricorrente con `tanh`; non usa la dinamica a costante di tempo variabile di `LTCNeuron`. Non ci sono ancora controlli completi sugli indici di vettori e matrici.
