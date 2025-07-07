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
    'v': 0.1, 
    'min_r': 0.02,    
    'max_r': 0.04
}



class impresa:
    def __init__(self, K, A, L, pi=0):
        self.K = K  # Capitale
        self.A = A # Patrimonio netto
        self.L = L  # Debito
        self.pi = pi # Profitto
        self.r = PARAMS['min_r'] 
class HIAModel:
    def __init__(self, NumeroImpreseIniziali=10000):
        self.NumeroImpreseIniziali = NumeroImpreseIniziali
       
        self.impresa = [impresa(K=100, A=20, L=80) for _ in range(self.NumeroImpreseIniziali)]
        self.PatrimonioNettoBanca = 100 #Et
        self.totaleCapitalePrecedente = sum(f.K for f in self.impresa)
        self.totalePatrimonioPrecedente = sum(f.A for f in self.impresa)
        self.TassoInteresseMedioPeriodoPrecedente = PARAMS['min_r'] 
        self.history = []
        self.y = 0 
        self.l = 0
        self.profittoBanca = 0


    def run_simulation(self, numeroPeriodi):
        for t in range(1, numeroPeriodi + 1):
           
            
            
            # gestione fallimenti con pulizia 
            bancarottaImpresa = [f for f in self.impresa if f.pi + f.A < 0]
            bruttoDebito = sum(max(0, f.L - f.K) for f in bancarottaImpresa)
            self.impresa = [f for f in self.impresa if f.pi + f.A >= 0]
            
            # Reset totali per il periodo corrente
            self.totaleCapitalePrecedente = sum(f.K for f in self.impresa)
            self.totalePatrimonioPrecedente = sum(f.A for f in self.impresa)
            self.l = 0  # Reset credito totale

            # Gestione nuovi entranti
            self.gestioneNuoveImprese()
            
            # Calcolo credito totale disponibile
            totaleDisponibilitàCredito = self.PatrimonioNettoBanca / PARAMS['v']
            
            # Ciclo imprese 
            for impresa in self.impresa:
                A_t_meno_1, K_t_meno_1, pi_t_meno_1 = impresa.A, impresa.K, impresa.pi

                # Allocazione credito
                kappa = K_t_meno_1 / self.totaleCapitalePrecedente if self.totaleCapitalePrecedente > 0 else 0
                alpha = A_t_meno_1 / self.totalePatrimonioPrecedente if self.totalePatrimonioPrecedente > 0 else 0
                L_it = totaleDisponibilitàCredito * (PARAMS['lambda'] * kappa + (1 - PARAMS['lambda']) * alpha)
                impresa.L = L_it

                # Calcolo tasso interesse 
                numeratore = 2 + A_t_meno_1
                denominatore = (2* PARAMS['c']* PARAMS['g']*(1/(PARAMS['phi']*PARAMS['c'])+ pi_t_meno_1+A_t_meno_1)+ 2*PARAMS['c']*PARAMS['g']*impresa.L)
                
                impresa.r = max(PARAMS['min_r'], min(PARAMS['max_r'], numeratore / denominatore))  
                # Calcolo capitale desiderato
                try:
                    term1 = (PARAMS['phi'] - PARAMS['g']*impresa.r)/(PARAMS['c']*PARAMS['phi']*PARAMS['g']*impresa.r)
                    term2 = A_t_meno_1/(2*PARAMS['g']*impresa.r)
                    K_it = term1 + term2  
                   
                except:
                    K_it = K_t_meno_1  # Mantiene capitale precedente in caso di errori
                
                impresa.K =  K_it
                impresa.A = impresa.K - impresa.L  # Patrimonio netto aggiornato
                # Patrimonio netto secondo pubblicazione
                '''(1/PARAMS['phi'])*(PARAMS['g']*impresa.r - (A_t_meno_1/K_it)) '''
                
                u_it = np.random.uniform(0, 2) # Simulazione di shock casuali con distribuzione uniforme da 0 a 2
                impresa.pi = (u_it * PARAMS['phi'] - PARAMS['g'] * impresa.r) * impresa.K if impresa.K > 0 else 0

            # Calcolo produzione aggregata come somma del capitale di tutte le imprese
        
            self.y = sum(f.K for f in self.impresa)

            # Calcolo tasso medio corretto usando media geometrica ponderata con i loans
            if self.impresa:
                # Prendiamo solo i dati del periodo corrente
                loans = np.array([f.L for f in self.impresa if hasattr(f, 'L')])
                tassi = np.array([f.r for f in self.impresa if hasattr(f, 'r')])
                
                mask = (loans > 0) & (tassi > 0)  # Aggiungiamo controllo positività
                if np.any(mask):
                    log_tassi = np.log(tassi[mask])
                    peso = loans[mask] / np.sum(loans[mask])
                    ponderazione = np.sum(log_tassi * peso)
                    self.TassoInteresseMedioPeriodoPrecedente = np.exp(ponderazione)
                else:
                    self.TassoInteresseMedioPeriodoPrecedente = PARAMS['min_r']

            # Aggiornamento credito totale dopo il ciclo
            self.l = sum(f.L for f in self.impresa)

            # Calcolo profitto banca 
            r_medio = self.TassoInteresseMedioPeriodoPrecedente
            D_t = totaleDisponibilitàCredito - self.PatrimonioNettoBanca
            interesseAttivo = sum(f.r * f.L for f in self.impresa)
            interessePassivo = r_medio * ((1 - PARAMS['omega']) * D_t + self.PatrimonioNettoBanca)
            self.profittoBanca = interesseAttivo - interessePassivo

            # Update patrimonio banca 
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
        
        numeroNuoviEntrati = int(round(probabilitàEntrata * N_bar))   
       
        if numeroNuoviEntrati == 0 or not self.impresa:
            return

        # Calcolo valori di riferimento per nuove imprese
        capitaleImpresePresenti = [f.K for f in self.impresa if f.K > 0]
        
        if not capitaleImpresePresenti:
            return

        
        # Capitale
        moda = stats.mode(capitaleImpresePresenti, keepdims=True).mode[0]

        # Creazione nuove imprese
        for _ in range(numeroNuoviEntrati):
            nuovoK = moda
            nuovoA = nuovoK * PARAMS['nu']  # Utilizziamo nu invece di modaEquity_ratio
            nuovoL = nuovoK - nuovoA
            self.impresa.append(impresa(K=nuovoK, A=nuovoA, L=nuovoL))

    def log_history(self, t):


        
            
        # Salva i dati della prima impresa (indice 0) se esiste almeno una impresa
        if len(self.impresa) > 0:
            impresa0 = self.impresa[0]
            impresa0_K = impresa0.K
            impresa0_A = impresa0.A
            impresa0_L = impresa0.L
            impresa0_pi = impresa0.pi
            impresa0_r = impresa0.r
        else:
            impresa0_K = impresa0_A = impresa0_L = impresa0_pi = impresa0_r = np.nan

        self.history.append({
            'periodo': t,
            'num_impresa': len(self.impresa),
            'produzione': self.y,
            'interesse medio': self.TassoInteresseMedioPeriodoPrecedente,
            'patrimonio netto banca': self.PatrimonioNettoBanca,
            'profitto banca': self.profittoBanca,
            'credito disponibile': 10 * self.PatrimonioNettoBanca,
            'Credito imprese ': self.l,
            # Dati impresa 0
            'impresa0_K': impresa0_K,
            'impresa0_A': impresa0_A,
            'impresa0_L': impresa0_L,
            'impresa0_pi': impresa0_pi,
            'impresa0_r': impresa0_r
        })
        
       
        print(self.history[-1])

if __name__ == "__main__":
    model = HIAModel()
    model.run_simulation(100)
    create_excel_report(model)
    df_results = pd.DataFrame(model.history)
    
    # Grafici migliorati
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    df_results['log_Y'] = np.log(df_results['produzione'])
    print(df_results[['periodo', 'log_Y', 'num_impresa', 'patrimonio netto banca', 'interesse medio']].head())
    # Grafico 1: Andamento della produzione aggregata (logaritmo del PIL) nel tempo
    axes[0,0].plot(df_results['periodo'], df_results['log_Y'], color='blue')
    axes[0,0].set_title('Log Produzione aggregata nel tempo')
    axes[0,0].set_xlabel('Periodo')
    axes[0,0].set_ylabel('log(Y aggregato)')
    axes[0,0].grid(True)
        
    # Grafico 2: Evoluzione del numero totale di imprese attive nel tempo
    axes[0,1].plot(df_results['periodo'], df_results['num_impresa'], color='orange')
    axes[0,1].set_title('Numero imprese nel tempo')
    axes[0,1].set_xlabel('Periodo')
    axes[0,1].set_ylabel('Numero imprese')
    
    # Grafico 3: Tasso di interesse medio applicato alle imprese nel tempo
    axes[1,0].plot(df_results['periodo'], df_results['interesse medio'], color='red')
    axes[1,0].set_title('Tasso di interesse medio nel tempo')
    axes[1,0].set_xlabel('Periodo')
    axes[1,0].set_ylabel('Tasso di interesse medio')
    axes[1,0].grid(True)
    
    # Grafico 4: Andamento del patrimonio netto della banca nel tempo
    axes[1,1].plot(df_results['periodo'], df_results['patrimonio netto banca'], color='green')
    axes[1,1].set_title('Patrimonio netto bancario nel tempo')
    axes[1,1].set_ylabel('Patrimonio netto banca')

    plt.tight_layout()
    plt.show()


