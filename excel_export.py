import pandas as pd
import os
from datetime import datetime

def create_excel_report(model, filename=None):
    # Create filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_report_{timestamp}.xlsx"
    
    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)
    
    # Create Excel writer
    writer = pd.ExcelWriter(filepath, engine='openpyxl')
    
    # Create history DataFrame
    df_history = pd.DataFrame(model.history)
    
    # Create current state DataFrames
    companies_data = [{
        'ID': i+1,
        'Capitale (K)': imp.K,
        'Patrimonio Netto (A)': imp.A,
        'Debito (L)': imp.L,
        'Profitto (π)': imp.pi,
        'Tasso Interesse (r)': imp.r
    } for i, imp in enumerate(model.impresa)]
    
    df_companies = pd.DataFrame(companies_data)
    
    # Summary statistics
    summary_data = {
        'Metriche': [
            'Numero Imprese',
            'Produzione Totale',
            'Capitale Totale',
            'Debito Totale',
            'Patrimonio Netto Banca',
            'Profitto Banca',
            'Credito Totale Disponibile'  # Aggiunta questa riga
        ],
        'Valori': [
            len(model.impresa),
            model.y,
            sum(f.K for f in model.impresa),
            sum(f.L for f in model.impresa),
            model.PatrimonioNettoBanca,
            model.profittoBanca,
            10 * model.PatrimonioNettoBanca  # Aggiunta questa riga
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    
    # Write to Excel
    df_history.to_excel(writer, sheet_name='Serie Storiche', index=False)
    df_companies.to_excel(writer, sheet_name='Stato Imprese', index=False)
    df_summary.to_excel(writer, sheet_name='Riepilogo', index=False)
    
    # Save and close
    writer.close()
    
    print(f"Report Excel salvato in: {filepath}")
    return filepath