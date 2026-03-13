# == Some notes ==
# Want to reproduce the results from https://arxiv.org/pdf/2411.05628 on using QC for (2+1)-QED

# External modules
from modules import *

# My modules
from classes import *
from running_helpers import *
from QED_hamiltonian import *
from my_spsa import *

parameters = {
    'L_x': 3,
    'L_y': 2,
    'gauge_truncation': 3,
    'n_fermion_layers': 2,
    'n_extra_layers': 0,
    'shots': 10000,
    'm': 1.0,
    'g': 0.3,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [((1,0),2), ((2,0),2)],
    'gauss_weight': 0.0, # Enforces Gauss' law
    'charge_weight': 0.0,
    'simulator': CPU_NOISELESS_SIMULATOR 
}

extra_parameters = {
    'sparse_test': False,
    'noisy': False,
    'diagnostics' : False,
    'print_circuit': False,
    'include_ps': False
    }
    
SPSA_parameters = {
    'max_iters' : 5000,
    'average_length' : 5,
    'grad_tol' : 1e-12,
    'average_tol' : 1e-10,
    'a' : 1.0,
    'c' : 0.5,
    'prints' : False,
    'diagnostics' : False
    }

print("RUNNING PLAYGROUND")

def Es_function_pg(qubits_per_gauge):
    I_string_list = list('I' * qubits_per_gauge)
    coeff = -0.5
    Es = {}
    
    for k in range(qubits_per_gauge - 1):
        temp_string = copy.copy(I_string_list)
        temp_string[k] = 'Z'
        Es["".join(temp_string)] = coeff * 2**(k)
        
    first_string = copy.copy(I_string_list)    
    first_string[qubits_per_gauge - 1] = 'Z'
    Es["".join(first_string)] = coeff * (2**(qubits_per_gauge - 1) - 1)
    return Es

qubits_per_gauge = int(np.ceil(np.log2(2*parameters['gauge_truncation']+1)))
hamiltonian = Hamiltonian(qubits_per_gauge)
hamiltonian.hamiltonian = Es_function_pg(parameters['gauge_truncation'])
print(hamiltonian.to_matrix())
