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
    'gauge_truncation': 1,
    'n_fermion_layers': 2,
    'n_extra_layers': 0,
    'shots': 10000,
    'm': 1.0,
    'g': 0.3,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [((1,0), 2), ((2,0), 2)],
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

print("RUNNING TEST")
print("parameters:")
for key in list(parameters):
    print(key,":",parameters[key])

for key in list(extra_parameters):
    print(key,":",extra_parameters[key])
    
for key in list(SPSA_parameters):
    print(key,":",SPSA_parameters[key])

parameters['g'] = 0.3
gs = []
V = []
V_nogauss = []

lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (0,0), anticharge_site = (2,1), E_0 = [0.0, 0.0])

for g in np.linspace(0.3, 3.0, 12):
    print("-"*10)
    parameters['g'] = g
    print("g =",g)
    circuit, thetas, total_thetas, n_fermion_thetas = initiate_circuit_observables_match_paper(parameters, lattice)
    measurer = Measurements_gpu(parameters['simulator'], lattice.n_qubits)
    
    hamiltonian = generate_qed_hamiltonian(parameters, lattice)
    guess = np.array([np.random.uniform(-0.08, 0.08) for _ in range(total_thetas)])
    
    def cost_function_qed(thetas_values):
        if extra_parameters['noisy']:
            cost = qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
        else:
            cost = qed_vqe_noiseless(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
        return cost
    
    results = NG_SPSA(cost_function_qed, guess, SPSA_parameters)
    
    best_thetas = results['final_paras']

    observes = observe_and_print_noiseless(thetas, best_thetas, circuit, lattice)
    
    final_energy = results['final_cost']
    print("Final cost:", final_energy)
    
    gs.append(g)
    V.append(final_energy)
    
print("gs=",gs)
print("V=",V)