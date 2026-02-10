# External modules
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
    'n_fermion_layers': 0,
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
measurer.build_measurement_error_mitigation(shots = 8192)

hamiltonian = generate_qed_hamiltonian(parameters, lattice)
thetas_values = [0.0]*total_thetas

ev = qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer)

print(ev)