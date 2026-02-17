# == Some notes ==
# Want to reproduce the results from https://arxiv.org/pdf/2411.05628 on using QC for (2+1)-QED

# External modules
from modules import *

# My modules
from classes import *
from running_helpers import *
from QED_hamiltonian import *

parameters = {
    'L_x': 2,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers': 20,
    'n_extra_layers': 4,
    'shots': 5000,
    'm': 0.5,
    'g': 1.0,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [],
    'gauss_weight': 1.0, # Enforces Gauss' law
    'charge_weight': 0.0,
    'simulator': CPU_NOISELESS_SIMULATOR 
}

extra_parameters = {
    'sparse_test': False,
    'noisy': False,
    'diagnostics' : False,
    'print_circuit': False,
    'include_ps': True
    }

print("RUNNING TEST")
print("parameters:")
for key in list(parameters):
    print(key,":",parameters[key])

for key in list(extra_parameters):
    print(key,":",extra_parameters[key])

parameters['g'] = 0.3
lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [0.0, 0.0])

if extra_parameters['print_circuit']: build_and_draw(parameters, lattice)

print("-"*8)
print("Circuit-based:")
thetas, energy_0 = fuller_runner(parameters, lattice, extra_parameters)

hamiltonian = generate_qed_hamiltonian(parameters, lattice)
H_sparse = hamiltonian.to_sparse_matrix()
eig_val, eig_vec = spar.linalg.eigsh(
    H_sparse,
    k=1,              # number of eigenvalues
    which='SA'        # Smallest Algebraic
)
classical_energy_0 = eig_val[0]
ground_vec = eig_vec[:, 0]
print("Matrix-based:")
observe_and_print_sparse(ground_vec, lattice)
print("Energy:",classical_energy_0)


lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (0,0), anticharge_site = (1,1), E_0 = [0.0, 0.0])
gs = []
gaps = []
gaps_classical = []
for g in np.linspace(0.3, 12.0, 10):
    print("-"*8)
    print("Circuit-based:")
    parameters['g'] = g
    thetas, energy_j = fuller_runner(parameters, lattice, extra_parameters)
    gs.append(g)
    gaps.append(energy_j - energy_0)
    
    hamiltonian = generate_qed_hamiltonian(parameters, lattice)
    H_sparse = hamiltonian.to_sparse_matrix()
    eig_val, eig_vec = spar.linalg.eigsh(
        H_sparse,
        k=1,              # number of eigenvalues
        which='SA'        # Smallest Algebraic
    )
    classical_energy_j = eig_val[0]
    ground_vec = eig_vec[:, 0]
    print("Matrix-based:")
    observe_and_print_sparse(ground_vec, lattice)
    print("Energy:",classical_energy_j)
    gaps_classical.append(classical_energy_j - classical_energy_0)

print("gs=",gs)
print("gaps=",gaps)
print("gaps_classical=",gaps_classical)