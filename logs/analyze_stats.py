
import pandas as pd
from scipy import stats
import sys

try:
    df = pd.read_csv('logs/training_metrics.csv')
    
    # Define phases
    phase1 = df[df['Epsilon'] > 0.5] # Exploration
    phase2 = df[df['Epsilon'] < 0.1] # Exploitation
    
    # Production T-test
    t_prod, p_prod = stats.ttest_ind(phase1['BusesProduced'], phase2['BusesProduced'], equal_var=False)
    
    # Efficiency T-test
    t_eff, p_eff = stats.ttest_ind(phase1['AvgEfficiency'], phase2['AvgEfficiency'], equal_var=False)
    
    print(f"Production Stats: Mean_P1={phase1['BusesProduced'].mean():.2f}, Mean_P2={phase2['BusesProduced'].mean():.2f}")
    print(f"Production T-Test: t={t_prod:.4f}, p={p_prod:.4e}")
    
    print(f"Efficiency Stats: Mean_P1={phase1['AvgEfficiency'].mean():.2f}, Mean_P2={phase2['AvgEfficiency'].mean():.2f}")
    print(f"Efficiency T-Test: t={t_eff:.4f}, p={p_eff:.4e}")

except Exception as e:
    print(f"Error: {e}")
