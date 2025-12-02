from modules import *
from classes import *
from circuit_helpers import *
from plot_helpers import *
from QED_hamiltonian import *

def smart_round(number, dec_places):
    re = 0.0 if abs(number.real) < 1e-18 else number.real
    im = 0.0 if abs(number.imag) < 1e-8  else number.imag

    re = round(re, dec_places)
    im = round(im, dec_places)

    if im == 0.0:
        return float(re)
    return complex(re, im)

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
    
def run_and_print_circuit(parameters, E_0_values):
    def run_and_print(E_0):
        parameters['E_0'] = E_0
    
        circuit, observables, thetas, total_thetas, n_qubits = initiate_circuit_observables(parameters)
        thetas_values = [np.random.uniform(0,1)]*total_thetas
    
        hamiltonian = generate_qed_hamiltonian(parameters)
        
        print("HAMILTONIAN GENERATED")
    
        def thetas_only_wrapper(thetas_values):
            cost = qed_vqe(thetas_values, thetas, hamiltonian, circuit, observables, parameters['shots'])
    
            return cost
    
        mini = scipy.optimize.minimize(thetas_only_wrapper, thetas_values, method = "COBYLA")
    
        print("VQE RUN")
        def get_state_counts(thetas_values, thetas, circuit, observables, n_qubits, shots):
            param_dict = dict(zip(thetas, thetas_values))
            circuit_values = circuit.assign_parameters(param_dict)
    
            return observables.full_z(circuit_values, n_qubits, shots)
    
        param_dict = dict(zip(thetas, mini.x))
        circuit_values = circuit.assign_parameters(param_dict)
        p_n = observables.particle_number(circuit_values, parameters['shots'])
        print("OBSERVABLE CALCULATED")
        print(f"P_n at E_0 = {E_0}: {p_n}")
        return p_n
    
    for E_0 in E_0_values:
        print('-'*30)
        run_and_print(E_0)

def run_and_print_sparse(parameters, E_0_values):
    lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'])
    P_n = particle_number_hamiltonian(lattice)
    charg_total = charge_total_hamiltonian(lattice)
    # charge_temp = Hamiltonian(lattice.n_qubits)
    # charge_temp.multiply_hamiltonians(charge_temp)
    # charge_temp.hamiltonian = multiply_hamiltonian_by_constant(charg_total.hamiltonian, 10.0)

    print("LATTICE AND OBSERVABLE HAMILTONIANS GENERATED")

    P_n_matrix = P_n.to_sparse_matrix()
    charg_total_matrix = charg_total.to_sparse_matrix()

    print("OBSERVABLE MATRICES GENERATED")

    def run_and_print(E_0):
        parameters['E_0'] = E_0

        hamiltonian = generate_qed_hamiltonian(parameters)
        
        # hamiltonian.add_hamiltonians(charge_temp)

        print("PROBLEM HAMILTONIAN GENERATED")

        matrix = hamiltonian.to_sparse_matrix()

        print("PROBLEM MATRIX GENERATED")

        vals, vecs = spar.linalg.eigsh(matrix, k=1)
        groundstate_vec = vecs[:, 0]
        groundstate_eig = vals[0]
        p_n = np.vdot(groundstate_vec, P_n_matrix @ groundstate_vec)
        total_charge = np.vdot(groundstate_vec, charg_total_matrix @ groundstate_vec)
        E_0 = smart_round(E_0, 5)
        p_n = smart_round(p_n, 5)
        total_charge = smart_round(total_charge, 5)
        print("OBSERVABLES CALCULATED")
        print(f"P_n at E_0 = {E_0}: {p_n}")
        print(f"total_charge at E_0 = {E_0}: {total_charge}")
        return p_n

    for E_0 in E_0_values:
        print('-'*30)
        run_and_print(E_0)