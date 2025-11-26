from modules import *
from classes import *
from circuit_helpers import *
from plot_helpers import *
from QED_hamiltonian import *

def static_potential_test(g, parameters):
    parameters['g'] = g

    circuit, observables, thetas, total_thetas, n_qubits = initiate_circuit_observables(parameters)
    thetas_values = [np.random.uniform(0,1)]*total_thetas

    hamiltonian = generate_qed_hamiltonian(parameters)

    def thetas_only_wrapper(thetas_values):
        cost = qed_vqe(thetas_values, thetas, hamiltonian, circuit, observables, parameters['shots'])
        print(cost)
        return cost

    mini = scipy.optimize.minimize(thetas_only_wrapper, thetas_values, method = "COBYLA")
    print(mini)

    def get_state_counts(thetas_values, thetas, circuit, observables, n_qubits, shots):
        param_dict = dict(zip(thetas, thetas_values))
        circuit_values = circuit.assign_parameters(param_dict)

        return observables.full_z(circuit_values, n_qubits, shots)

    counts = get_state_counts(mini.x, thetas, circuit, observables, n_qubits, 512)
    print("Counts with shots = 512: ")
    print(counts)
    param_dict = dict(zip(thetas, mini.x))
    circuit_values = circuit.assign_parameters(param_dict)
    energy = observables.energy(circuit_values, hamiltonian, parameters['shots'])
    print(f"Energy at g = {g}: {energy}")
    return energy