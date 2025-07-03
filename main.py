import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from excel_export import create_excel_report
PARAMS = {
    'phi': 0.1,
    'c': 1.0,
    'g': 1.1,
    'nu': 0.08,
    'omega': 0.002,
    'lambda': 0.3,
    'd': 100.0,
    'e': 0.1,
    'N_bar': 180,
    'v': 0.1,  # parametro di leverage bancario  # Numero massimo di imprese
    'min_r': 0.02,     # Aumentato il tasso minimo
    'max_r': 0.08    # Massimo capitale per impresa
}



class impresa:
    def __init__(self, K, A, L, pi=0):
        self.K = K  # Capitale
        self.A = A # Patrimonio netto
        self.L = L  # Debito
        self.pi = pi # Profitto
        self.r = 0 # Tasso di interesse

class HIAModel:
    def __init__(self, NumeroImpreseIniziali=10000):
        self.NumeroImpreseIniziali = NumeroImpreseIniziali
        self.impresa = [impresa(K=100, A=20, L=80) for _ in range(self.NumeroImpreseIniziali)]
        self.PatrimonioNettoBanca = 1000 #Et
        self.totaleCapitalePrecedente = sum(f.K for f in self.impresa)
        self.totalePatrimonioPrecedente = sum(f.A for f in self.impresa)
        self.TassoInteresseMedioPeriodoPrecedente = 0 # Initialize with small positive value
        self.history = []
        self.y = 0 
        self.l = 0
        self.profittoBanca = 0

    def create_new_bank(self):
        """Creates a new bank with initial capital when the existing bank fails"""
        initial_capital = max(1000, self.PatrimonioNettoBanca * 0.5)  # Keep some existing capital if possible
        self.PatrimonioNettoBanca = initial_capital
        self.TassoInteresseMedioPeriodoPrecedente = 0
        self.profittoBanca = 0
        print(f"Nuova banca creata con capitale iniziale {initial_capital}")

    def run_simulation(self, numeroPeriodi):
        for t in range(1, numeroPeriodi + 1):
            # Controllo patrimonio banca
           
            # Gestione fallimenti secondo flowchart
            bancarottaImpresa = [f for f in self.impresa if f.pi + f.A < 0]
            bruttoDebito = sum(max(0, f.L - f.K) for f in bancarottaImpresa)
            self.impresa = [f for f in self.impresa if f.pi + f.A >= 0]

            # Gestione nuovi entranti
            self.gestioneNuoveImprese()
            
            # Calcolo credito totale disponibile secondo flowchart
            totaleDisponibilitàCredito = self.PatrimonioNettoBanca / PARAMS['v']
            
            # Update totali
            self.totaleCapitalePrecedente = sum(f.K for f in self.impresa)
            self.totalePatrimonioPrecedente = sum(f.A for f in self.impresa)

            # Ciclo imprese secondo flowchart
            for impresa in self.impresa:
                A_t_meno_1, K_t_meno_1, pi_t_meno_1 = impresa.A, impresa.K, impresa.pi

                # Allocazione credito secondo flowchart
                kappa = K_t_meno_1 / self.totaleCapitalePrecedente if self.totaleCapitalePrecedente > 0 else 0
                alpha = A_t_meno_1 / self.totalePatrimonioPrecedente if self.totalePatrimonioPrecedente > 0 else 0
                L_it = totaleDisponibilitàCredito * (PARAMS['lambda'] * kappa + (1 - PARAMS['lambda']) * alpha)
                impresa.L = L_it

                # Tasso interesse secondo la formula esatta
                numeratore = 2 + A_t_meno_1
                denominatore = (
                    2 * PARAMS['c'] * PARAMS['g'] * (1/(PARAMS['phi'] * PARAMS['c']) + pi_t_meno_1 + A_t_meno_1) +
                    2 * PARAMS['c'] * PARAMS['g'] * L_it * (PARAMS['lambda'] * kappa + (1 - PARAMS['lambda']) * alpha)
                )
                impresa.r = numeratore / denominatore if denominatore != 0 else PARAMS['max_r']
                impresa.r = max(PARAMS['min_r'], min(impresa.r, PARAMS['max_r']))

                # Calcolo capitale desiderato corretto
                try:
                    term1 = (PARAMS['phi'] - PARAMS['g']*impresa.r)/(PARAMS['c']*PARAMS['phi']*PARAMS['g']*impresa.r)
                    term2 = A_t_meno_1/(2*PARAMS['g']*impresa.r)
                    K_d = min(term1 + term2, PARAMS['max_K'])  # Limite massimo al capitale
                    K_d = min(K_d, 5 * A_t_meno_1)  # Limite basato su equity
                except:
                    K_d = K_t_meno_1  # Mantiene capitale precedente in caso di errori
                
                impresa.K = max(0, K_d)

                # Profitto secondo flowchart
                u_it = np.random.uniform(0, 2)
                impresa.pi = (u_it * PARAMS['phi'] - PARAMS['g'] * impresa.r) * impresa.K if impresa.K > 0 else 0

            # Calcolo produzione aggregata e aggiornamento
            self.y = sum(PARAMS['phi'] * f.K for f in self.impresa)
            self.l = sum(f.L for f in self.impresa)  # Total credit

            # Calcolo tasso medio corretto
            if self.impresa:
                tassi = [f.r for f in self.impresa if f.r > 0]
                self.TassoInteresseMedioPeriodoPrecedente = np.mean(tassi) if tassi else 0.01
            
            # Update patrimonio netto
            for impresa in self.impresa:
                impresa.A = impresa.A + impresa.pi  # Era sottratto invece che sommato!

            # Calcolo profitto banca secondo flowchart
            r_medio = np.mean([f.r for f in self.impresa]) if self.impresa else 0.01
            D_t = totaleDisponibilitàCredito - self.PatrimonioNettoBanca
            interesseAttivo = sum(f.r * f.L for f in self.impresa)
            interessePassivo = r_medio * ((1 - PARAMS['omega']) * D_t + self.PatrimonioNettoBanca)
            self.profittoBanca = interesseAttivo - interessePassivo

            # Update patrimonio banca secondo flowchart
            self.PatrimonioNettoBanca = self.profittoBanca + self.PatrimonioNettoBanca - bruttoDebito

            # Logging
            self.log_history(t)

            if t % 100 == 0:
                print(f"Periodo {t}/{numeroPeriodi} completato. Imprese attive: {len(self.impresa)}")

    def gestioneNuoveImprese(self):
        

        # Calcola probabilità entrata solo se tasso interesse non è troppo alto
        if self.TassoInteresseMedioPeriodoPrecedente > PARAMS['max_r']:
            return

        d, e, N_bar = PARAMS['d'], PARAMS['e'], PARAMS['N_bar']
        probabilitàEntrata = 1 / (1 + np.exp(d * (self.TassoInteresseMedioPeriodoPrecedente - e)))
        try:
            numeroNuoviEntrati = int(round(probabilitàEntrata * N_bar))   
        except ValueError:
            numeroNuoviEntrati = 0
        if numeroNuoviEntrati == 0 or not self.impresa:
            return

        # Calcolo valori di riferimento per nuove imprese
        capitaleImpresePresenti = [f.K for f in self.impresa if f.K > 0]
        equityRatio = [f.A / f.K for f in self.impresa if f.K > 0 and f.A > 0]
        
        if not capitaleImpresePresenti or not equityRatio:
            return

        
            # Capitale
        moda = stats.mode(capitaleImpresePresenti, keepdims=True).mode[0]

        # Equity ratio
        modaEquity_ratio = stats.mode(equityRatio, keepdims=True).mode[0]

        # Applica i vincoli come già facevi
        # Creazione nuove imprese
        for _ in range(numeroNuoviEntrati):
            nuovoK = moda
            nuovoA = nuovoK * modaEquity_ratio
            nuovoL = nuovoK - nuovoA
            self.impresa.append(impresa(K=nuovoK, A=nuovoA, L=nuovoL))

    def log_history(self, t):


        
            
        self.history.append({
            'periodo': t,
            'num_impresa': len(self.impresa),
            'produzione':self.y,
            'interesse medio': self.TassoInteresseMedioPeriodoPrecedente,
            'patrimonio netto banca': self.PatrimonioNettoBanca,
            'profitto banca': self.profittoBanca,
            'credito disponibile': 10 * self.PatrimonioNettoBanca,
            'Credito imprese ': self.l
        })
        
        # Stampa solo ogni 100 periodi per non sovraccaricare l'output
        print(self.history[-1])

if __name__ == "__main__":
    model = HIAModel()
    model.run_simulation(1000)
    create_excel_report(model)
    df_results = pd.DataFrame(model.history)
    
    # Grafici migliorati
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    df_results['log_Y'] = np.log(df_results['produzione'])
    print(df_results[['periodo', 'log_Y', 'num_impresa', 'patrimonio netto banca', 'interesse medio']].head())
    # Grafico 1: PIL nel tempo
    axes[0,0].plot(df_results['periodo'], df_results['log_Y'], color='blue')
    axes[0,0].set_title('Log Produzione aggregata nel tempo')
    axes[0,0].set_xlabel('Periodo')
    axes[0,0].set_ylabel('log(Y aggregato)')
    axes[0,0].grid(True)
        
    # Grafico 2: Numero imprese
    axes[0,1].plot(df_results['periodo'], df_results['num_impresa'], color='orange')
    axes[0,1].set_title('Numero imprese nel tempo')
    axes[0,1].set_xlabel('Periodo')
    axes[0,1].set_ylabel('Numero imprese')
    
    # Grafico 3: Equity bancario
    axes[1,0].plot(df_results['periodo'], df_results['interesse medio'], color='red')
    axes[1,0].set_title('Tasso di interesse medio nel tempo')
    axes[1,0].set_xlabel('Periodo')
    axes[1,0].set_ylabel('Tasso di interesse medio')
    axes[1,0].grid(True)
    
    # Grafico 4: Tasso di interesse medio
    axes[1,1].plot(df_results['periodo'], df_results['patrimonio netto banca'], color='green')
    axes[1,1].set_title('Patrimonio netto bancario nel tempo')
    axes[1,1].set_ylabel('Patrimonio netto banca')

    plt.tight_layout()
    plt.show()


