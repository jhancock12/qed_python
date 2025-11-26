from modules import *
from classes import *

def initiate_circuit_observables(parameters):
    L_x = parameters['L_x']
    L_y = parameters['L_y']
    n_fermion_layers = parameters['n_fermion_layers']
    gauge_truncation = parameters['gauge_truncation']
    dynamical_links = parameters['dynamical_links']
    measurer = Measurements()
    lattice = Lattice(L_x,L_y,gauge_truncation,dynamical_links)
    observables = ObservableCalculator(lattice,measurer)    
    
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_links*lattice.qubits_per_gauge)
    builder.initialize_fermions()
    n_slice = builder.iSwap_block_calculate()
    
    total_thetas = n_slice*n_fermion_layers + lattice.qubits_per_gauge*lattice.n_dynamical_links
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_slice*n_fermion_layers]
    gauge_thetas = thetas[n_slice*n_fermion_layers:]
    
    for j in range(n_fermion_layers):
        builder.iSwap_block(fermion_thetas[n_slice*j:n_slice*j+(n_slice)])
    builder.gauge_block(gauge_thetas, gauge_truncation)

    circuit = builder.build()

    return circuit, observables, thetas, total_thetas, lattice.n_qubits

def qed_vqe(thetas_values, thetas, hamiltonian, circuit, observables, shots):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)

    return observables.energy(circuit_values, hamiltonian, shots)