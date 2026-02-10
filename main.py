# == Some notes ==
# Want to reproduce the results from https://arxiv.org/pdf/2411.05628 on using QC for (2+1)-QED

# External modules
import QED_hamiltonian
from modules import *

# My modules
from classes import *
from circuit_helpers import *
from plot_helpers import *
from QED_hamiltonian import *
from running_helpers import *

parameters = {
    'L_x': 2,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers': 10,
    'shots': 10000,
    'm': 0.5,
    'g': 1.0,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [],
    'gauss_weight': 1.0, # Enforces Gauss' law
    'charge_weight': 0.0,
    'simulator': CPU_NOISELESS_SIMULATOR 
}

print("RUNNING TEST")
print("parameters:")
for key in list(parameters):
    print(key,":",parameters[key])
# General parameters

mass_multi = 1
electric_multi = 1
magnetic_multi = 1
kinetic_multi = 1

gs_classical = []
gaps_classical = []
gs_quantum = []
gaps_quantum = []

lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [0.0, 0.0])

circuit, observables, thetas, total_thetas, lattice.n_qubits = initiate_circuit_observables(parameters, lattice)

measurer = Measurements_gpu(parameters['simulator'], lattice.n_qubits)
# measurer.build_measurement_error_mitigation(shots = 8192)

hamiltonian = generate_qed_hamiltonian(parameters, lattice)

H_sparse = hamiltonian.to_sparse_matrix()
eig_val, eig_vec = spar.linalg.eigsh(
    H_sparse,
    k=1,              # number of eigenvalues
    which='SA'        # Smallest Algebraic
)
final_classical_value = eig_val[0]
print("True groundstate energy:",final_classical_value)

def thetas_only_wrapper(thetas_values):
    cost = qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer)
    return cost

guess = [np.random.uniform(0,1) for _ in range(total_thetas)]

mini = scipy.optimize.minimize(thetas_only_wrapper, guess, method = "COBYLA")

final_quantum_value = thetas_only_wrapper(mini.x)

print("Found groundstate energy:",final_quantum_value)
print("True groundstate energy:",final_classical_value)