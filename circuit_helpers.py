from modules import *
from classes import *

def initiate_circuit_observables(parameters, lattice):
    measurer = Measurements_gpu(parameters['simulator'])
    
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_gauge_qubits)
    builder.initialize_fermions(lattice)
    n_slice = builder.iSwap_block_calculate_qed(lattice, parameters['n_fermion_layers'])
    
    thetas_per_gauge = {
        2: 2,
        3: 4
    }
    
    total_thetas = n_slice * parameters['n_fermion_layers'] + thetas_per_gauge[lattice.qubits_per_gauge]*lattice.n_dynamical_links
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_slice*parameters['n_fermion_layers']]
    gauge_thetas = thetas[n_slice*parameters['n_fermion_layers']:]
    for j in range(parameters['n_fermion_layers']):
        builder.iSwap_block_qed(fermion_thetas[n_slice*j:n_slice*j+(n_slice)], lattice)
    builder.gauge_block(gauge_thetas, parameters['gauge_truncation'])

    circuit = builder.build()
    # print(circuit.draw())

    return circuit, thetas, total_thetas, len(fermion_thetas)

def qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    ev = measurer.expected_value_hamiltonian_qed(hamiltonian, circuit_values, lattice, shots)
    return ev
    
def qed_vqe_noiseless(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)

    circuit_nom = circuit_values.remove_final_measurements(inplace=False)
    psi = qiskit.quantum_info.Statevector.from_instruction(circuit_nom)

    H_qiskit = hamiltonian.to_qiskit()

    ev = np.real(psi.expectation_value(H_qiskit))
    return ev
    
def full_runner(thetas, circuit, hamiltonian, lattice, measurer, parameters, sparse_test = False, noisy = False, diagnostics = False):
    if sparse_test:
        print('='*10)
        H_sparse = hamiltonian.to_sparse_matrix()
        eig_val, eig_vec = spar.linalg.eigsh(
            H_sparse,
            k=1,              # number of eigenvalues
            which='SA'        # Smallest Algebraic
        )
        final_classical_value = eig_val[0]
        ground_vec = eig_vec[:, 0]
        print("True groundstate energy:",final_classical_value)
        observe_and_print_sparse(ground_vec, lattice)
        print('='*10)

    def gauge_function(thetas_values):
        thetas_values_full = [0.0]*n_fermion_thetas + list(thetas_values)
        
        if noisy:
            cost = qed_vqe(thetas, thetas_values_full, circuit, hamiltonian, lattice, measurer, parameters['shots'])
        else:
            cost = qed_vqe_noiseless(thetas, thetas_values_full, circuit, hamiltonian, lattice, measurer, parameters['shots'])
            
        if diagnostics:
            print('-'*10)
            print("Energy:",cost)
            observe_and_print_circuit(thetas, thetas_values_full, circuit, lattice, measurer, parameters['shots'])
        return cost
    
    def full_function(thetas_values):
        if noisy:
            cost = qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
        else:
            cost = qed_vqe_noiseless(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
            
        if diagnostics:
            print('-'*10)
            print("Energy:",cost)
            observe_and_print_circuit(thetas, thetas_values, circuit, lattice, measurer, parameters['shots'])
        return cost
        
    def parameter_shift_gradient(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
        grads = np.array([0.0]*len(thetas_values))
        for k in range(len(thetas_values)):
            thetas_plus  = np.array(thetas_values, copy=True)
            thetas_minus = np.array(thetas_values, copy=True)
    
            thetas_plus[k] += np.pi/2 
            thetas_minus[k] -= np.pi/2
            
            if noisy:
                cost_plus = qed_vqe(thetas, thetas_plus, circuit, hamiltonian, lattice, measurer, shots)
                cost_minus = qed_vqe(thetas, thetas_minus, circuit, hamiltonian, lattice, measurer, shots)
            else:
                cost_plus = qed_vqe_noiseless(thetas, thetas_plus, circuit, hamiltonian, lattice, measurer, shots)
                cost_minus = qed_vqe_noiseless(thetas, thetas_minus, circuit, hamiltonian, lattice, measurer, shots)
                
            grads[k] = 0.5 * (cost_plus - cost_minus)
        return grads
    
    def grad_function(thetas_values):
        grad = parameter_shift_gradient(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, 2*parameters['shots'])
        return grad
        
    print('='*10)
    guess_gauge = np.array([np.pi, 0.0]*int(n_gauge_thetas/2)) + np.random.uniform(-0.08, 0.08, size=n_gauge_thetas)
    mini = scipy.optimize.minimize(gauge_function, guess_gauge, method = "COBYLA")
    final_gauge_thetas = mini.x
    final_gauge_value = gauge_function(final_gauge_thetas)
    print("Found groundstate energy after gauge part (COBYLA):", final_gauge_value)
    print('='*10)
    
    print('='*10)
    guess_fermion = np.random.uniform(0, 0.08, size=n_fermion_thetas)
    guess_full = list(guess_fermion) + list(final_gauge_thetas)
    mini = scipy.optimize.minimize(full_function, guess_full, method = "COBYLA")
    final_thetas = mini.x
    final_value = full_function(final_thetas)
    print("Found groundstate energy after fermion part (COBYLA):", final_value)
    print('='*10)
    
    print('='*10)
    mini = scipy.optimize.minimize(
        fun=full_function,
        jac=grad_function,
        x0=final_thetas,
        method="L-BFGS-B",
    )
    final_final_thetas = mini.x
    final_quantum_value = full_function(mini.x)
    print("Found groundstate energy after parameter-shift finalization:", final_quantum_value)
    print('='*10)

def fuller_runner(parameters, lattice, sparse_test = False, noisy = False, diagnostics = False):
    circuit, thetas, total_thetas, n_fermion_thetas = initiate_circuit_observables(parameters, lattice)
    n_gauge_thetas = total_thetas - n_fermion_thetas
    
    measurer = Measurements_gpu(parameters['simulator'], lattice.n_qubits)
    
    hamiltonian = generate_qed_hamiltonian(parameters, lattice)
    
    full_runner(thetas, circuit, hamiltonian, lattice, measurer, parameters, sparse_test, noisy, diagnostics)