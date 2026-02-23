# Fair Sequence Allocation — Metaheuristics (Python)

Questo progetto implementa una variante ispirata al paper *Fair resource-constrained allocation of task-chains* fileciteturn0file0,
con vincoli di:
- **precedenza** (task in catena per agente, ordine non decrescente sui time slot)
- **capacità** per time slot (risorsa non-resumable: ogni slot ha una capacità fissa)

e con:
- **soluzioni iniziali**: round-robin, casuale, greedy-fair
- **ricerca locale**: mosse di **spostamento** (shift di boundary tra slot) e **scambio** (swap di boundary tasks)
- metaeuristiche: **Simulated Annealing**, **Tabu Search**, **Genetic Algorithm** (+ local search puro)

## Struttura dataset attesa

Metti le istanze in `data/` così (come descritto nel readme della prof):

```
data/
  class_1/
    1/
      I_1_size.csv
      I_1_capacities.csv
      I_1_requirements.csv
    2/
      ...
  class_2/
    ...
```

Ogni CSV usa `;` come separatore e non ha header.

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Esecuzione rapida (su una singola istanza)

```bash
python -m src.cli run-instance --instance-dir data/class_1/1 --algo sa --time-limit 180 --seed 1
```

## Batch esperimenti (tutte le istanze e più algoritmi)

```bash
python -m src.cli run-batch --data-root data --time-limits 180 300 600 --seeds 1 2 3
```

Output in `results/`:
- `results/raw_results.csv` (tutte le run)
- `results/summary.csv` (media/mediana per algoritmo×classe×time-limit)
- `results/best_by_instance.csv` (miglior algoritmo per ogni istanza)

## Obiettivi (metriche)
Per una soluzione ϑ:
- **Fairness objective**: `max_j f_j(ϑ)` dove `f_j` è la *media* degli indici di slot assegnati ai task dell'agente j (slot 1..T).
- **System cost**: `sum_j f_j(ϑ)`.

Lato esperimenti puoi confrontare:
- miglior valore fairness (più basso = più equo)
- system cost (efficienza globale)
- *price of fairness* rispetto al migliore system cost osservato (approssimato, perché qui non risolviamo MILP ottimo)


## Dashboard (Flask)

Avvio:

```bash
python -m src.dashboard.app
```

Poi apri: http://127.0.0.1:5000

La dashboard permette di:
- scegliere dataset/classe/istanza
- scegliere algoritmo (ls/sa/tabu/ga), inizializzazione, time limit, seed
- lanciare run in background e vedere la curva best-so-far (max-avg completion vs tempo)
- confrontare le combinazioni migliori nella pagina “Compare” usando le run salvate dalla dashboard

Le run vengono salvate in `results/dashboard_runs/run_<id>.json`.
