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
    'n_fermion_layers': 5,
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

lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [0.0, 0.0])

circuit, thetas, total_thetas, lattice.n_qubits = initiate_circuit_observables(parameters, lattice)

measurer = Measurements_gpu(parameters['simulator'], lattice.n_qubits)
# measurer.build_measurement_error_mitigation(shots = 8192)

hamiltonian = generate_qed_hamiltonian(parameters, lattice)

def thetas_only_wrapper(thetas_values):
    print('-'*10)
    # print(circuit.draw())
    cost = qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, 2000)
    observe_and_print_circuit(thetas, thetas_values, circuit, lattice, measurer, 2000)
    return cost

def parameter_shift_gradient(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
    grads = np.array([0.0]*len(thetas_values))
    for k in range(len(thetas_values)):
        thetas_plus  = np.array(thetas_values, copy=True)
        thetas_minus = np.array(thetas_values, copy=True)

        thetas_plus[k] += np.pi/2
        thetas_minus[k] -= np.pi/2

        cost_plus = qed_vqe(thetas, thetas_plus, circuit, hamiltonian, lattice, measurer, shots)
        cost_minus = qed_vqe(thetas, thetas_minus, circuit, hamiltonian, lattice, measurer, shots)

        grads[k] = 0.5 * (cost_plus - cost_minus)
    return grads

def thetas_only_grad(thetas_values):
    grad = parameter_shift_gradient(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, 10000)
    return grad

guess = np.random.uniform(0, 0.1, size=total_thetas)
bounds = [(0, 2*np.pi)] * total_thetas

mini = scipy.optimize.minimize(
    fun=thetas_only_wrapper,
    jac=thetas_only_grad,
    x0=guess,
    method="L-BFGS-B",
    bounds=bounds
)

final_quantum_value = thetas_only_wrapper(mini.x)

print("Found groundstate energy:",final_quantum_value)

# Okay, I now want to change optimizer to something more modern and effective!
# Something that uses parameter-shift would be nice, or similarly something like SPSA
