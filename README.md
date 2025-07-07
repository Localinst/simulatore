# HIA Model Simulation

Questo progetto implementa una simulazione di un modello macroeconomico agent-based (HIA Model) in Python, dove imprese e banca interagiscono tramite credito, produzione e fallimenti.

## Requisiti

Assicurati di avere Python 3.8 o superiore installato.  
Installa le librerie necessarie con:

```bash
pip install numpy scipy matplotlib pandas seaborn openpyxl
```

> **Nota:**  
> Il modulo `excel_export` deve essere presente nella stessa cartella del progetto e deve contenere la funzione `create_excel_report(model)`.

## Come usare il progetto

1. **Clona o scarica la cartella del progetto.**
2. **Assicurati che tutti i file siano nella stessa directory:**
   - `main.py`
   - `excel_export.py` (o altro modulo per l'export Excel)
3. **Installa le dipendenze** (vedi sopra).
4. **Esegui la simulazione:**

```bash
python main.py
```

## Cosa fa il programma

- Simula l'evoluzione di un sistema di imprese e una banca per un certo numero di periodi.
- Registra, per ogni periodo, indicatori aggregati (produzione, numero imprese, patrimonio banca, ecc.) e l'andamento della prima impresa.
- Esporta i risultati in un file Excel.
- Mostra grafici sull'andamento di produzione, numero imprese, tasso di interesse medio e patrimonio netto bancario.

## Output

- **File Excel**: contiene la storia della simulazione, inclusi i dati della prima impresa.
- **Grafici**: vengono visualizzati automaticamente al termine della simulazione.

## Personalizzazione

Puoi modificare i parametri del modello cambiando i valori nel dizionario `PARAMS` all'inizio di `main.py`.

## Contatti

Per domande o suggerimenti, apri una issue.
