from modules import *
from classes import *

def initiate_circuit_observables(parameters, lattice):
    measurer = Measurements_gpu(parameters['simulator'])
    observables = ObservableCalculator(lattice, measurer)    
    
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_gauge_qubits)
    builder.initialize_fermions()
    n_slice = builder.iSwap_block_calculate()
    
    thetas_per_gauge = {
        2: 2,
        3: 4
    }
    
    total_thetas = n_slice * parameters['n_fermion_layers'] + thetas_per_gauge[lattice.qubits_per_gauge]*lattice.n_dynamical_links
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_slice*parameters['n_fermion_layers']]
    gauge_thetas = thetas[n_slice*parameters['n_fermion_layers']:]
    for j in range(parameters['n_fermion_layers']):
        builder.iSwap_block(fermion_thetas[n_slice*j:n_slice*j+(n_slice)])
    builder.gauge_block(gauge_thetas, parameters['gauge_truncation'])

    circuit = builder.build()
    print(circuit.draw())

    return circuit, observables, thetas, total_thetas, lattice.n_qubits

def qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    ev = measurer.expected_value_hamiltonian_qed(hamiltonian, circuit_values, lattice, shots = 1024)
    return ev