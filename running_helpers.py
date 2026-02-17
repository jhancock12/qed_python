from modules import *
from classes import *
from QED_hamiltonian import *

def smart_round(number, dec_places):
    if isinstance(number, dict):
        for key in list(number):
            number[key] = smart_round(number[key], dec_places)
        return number
    elif isinstance(number, list) or isinstance(number, np.ndarray):
        for k in range(len(number)):
            number[k] = smart_round(number[k], dec_places)
        return number
    else:
        re = 0.0 if abs(number.real) < 1e-8 else number.real
        im = 0.0 if abs(number.imag) < 1e-8  else number.imag

        re = round(re, dec_places)
        im = round(im, dec_places)

        if im == 0.0:
            return float(re)
        return complex(re, im)

def electric_field_values(ground_vec, lattice):
    electric_field_values_dict = {}
    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            electric_hamiltonian = Hamiltonian(lattice.n_qubits)
            electric_hamiltonian = electric_field_quadratic_term_n_direction(electric_hamiltonian, lattice, n, direction, lattice.dynamical_links_list)
            electric_matrix = electric_hamiltonian.to_matrix()

            value = np.vdot(ground_vec, electric_matrix @ ground_vec)
            electric_field_values_dict[(lattice.labels[n],direction)] = value
    return electric_field_values_dict

def magnetic_field_values(ground_vec, lattice):
    magnetic_field_values_dict = {}
    for n in lattice.plaquettes:
        magnetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        magnetic_hamiltonian = magnetic_term_n(magnetic_hamiltonian, lattice, n, lattice.dynamical_links_list)
        magnetic_matrix = magnetic_hamiltonian.to_matrix()
        value = np.vdot(ground_vec, magnetic_matrix @ ground_vec)
        magnetic_field_values_dict[lattice.labels[n]] = value
    return magnetic_field_values_dict

def total_charge(ground_vec, lattice):
    hamiltonian = charge_total_hamiltonian(lattice)
    matrix = hamiltonian.to_matrix()
    return np.vdot(ground_vec, matrix @ ground_vec)

def particle_number_values(ground_vec, lattice):
    particle_number_values_dict = {}
    total_pn = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            pn_n_matrix = pn_n.to_matrix()
            value = np.vdot(ground_vec, pn_n_matrix @ ground_vec)
            particle_number_values_dict[(x,y)] = value

    return particle_number_values_dict

def total_particle_number(ground_vec, lattice):
    total_pn = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            total_pn.add_hamiltonians(pn_n)
    total_matrix = total_pn.to_matrix()
    value = np.vdot(ground_vec, total_matrix @ ground_vec)
    return value

def gauss_law_values(ground_vec, lattice):
    gauss_law_values_dict = {}
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            n = lattice.reverse_labels[(x, y)]
            g_n = gauss_operator_n(lattice, n)
            g_n_matrix = g_n.to_matrix()
            value = np.vdot(ground_vec, g_n_matrix @ ground_vec)
            gauss_law_values_dict[(x,y)] = value

    return gauss_law_values_dict

def total_gauss_law(ground_vec, lattice):
    hamiltonian = gauss_hamiltonian_linear(lattice)
    matrix = hamiltonian.to_matrix()
    value = np.vdot(ground_vec, matrix @ ground_vec)
    return value

def observe_and_print(ground_vec, lattice):
    electric_field_values_dict = electric_field_values(ground_vec, lattice)
    magnetic_field_values_dict = magnetic_field_values(ground_vec, lattice)
    particle_number_value_dict = particle_number_values(ground_vec, lattice)
    gauss_law_values_dict = gauss_law_values(ground_vec, lattice)
    _total_particle_number = total_particle_number(ground_vec, lattice)
    _total_charge = total_charge(ground_vec, lattice)
    _total_gauss_law = total_gauss_law(ground_vec, lattice)

    electric_field_values_dict = smart_round(electric_field_values_dict, 6)
    magnetic_field_values_dict = smart_round(magnetic_field_values_dict, 6)
    particle_number_value_dict = smart_round(particle_number_value_dict, 6)
    gauss_law_values_dict = smart_round(gauss_law_values_dict, 6)
    _total_particle_number = smart_round(_total_particle_number, 6)
    _total_charge = smart_round(_total_charge, 6)
    _total_gauss_law = smart_round(_total_gauss_law, 6)
    
    print("electric_field_values_dict:",electric_field_values_dict)
    print("magnetic_field_values_dict:",magnetic_field_values_dict)
    print("particle_number_value_dict:",particle_number_value_dict)
    print("gauss_law_values_dict (no charges):",gauss_law_values_dict)
    print("total_charge:",_total_charge)
    print("total_gauss (with charges):",_total_gauss_law)
    print("total_particle_number:",_total_particle_number)
    
def electric_field_values_sparse(ground_vec, lattice):
    electric_field_values_dict = {}

    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            electric_hamiltonian = Hamiltonian(lattice.n_qubits)
            electric_hamiltonian = electric_field_quadratic_term_n_direction(
                electric_hamiltonian, lattice, n, direction, lattice.dynamical_links_list
            )

            electric_matrix = electric_hamiltonian.to_sparse_matrix()
            value = np.vdot(ground_vec, electric_matrix @ ground_vec)

            electric_field_values_dict[(lattice.labels[n], direction)] = value

    return electric_field_values_dict

def magnetic_field_values_sparse(ground_vec, lattice):
    magnetic_field_values_dict = {}

    for n in lattice.plaquettes:
        magnetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        magnetic_hamiltonian = magnetic_term_n(
            magnetic_hamiltonian, lattice, n, lattice.dynamical_links_list
        )

        magnetic_matrix = magnetic_hamiltonian.to_sparse_matrix()
        value = np.vdot(ground_vec, magnetic_matrix @ ground_vec)

        magnetic_field_values_dict[lattice.labels[n]] = value

    return magnetic_field_values_dict

def total_charge_sparse(ground_vec, lattice):
    hamiltonian = charge_total_hamiltonian(lattice)
    matrix = hamiltonian.to_sparse_matrix()
    return np.vdot(ground_vec, matrix @ ground_vec)

def particle_number_values_sparse(ground_vec, lattice):
    particle_number_values_dict = {}

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            pn_n_matrix = pn_n.to_sparse_matrix()

            value = np.vdot(ground_vec, pn_n_matrix @ ground_vec)
            particle_number_values_dict[(x, y)] = value

    return particle_number_values_dict

def total_particle_number_sparse(ground_vec, lattice):
    total_pn = Hamiltonian(lattice.n_qubits)

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            sign = (-1)**(x+y)
            pn_n.hamiltonian = multiply_hamiltonian_by_constant(pn_n.hamiltonian, sign)
            total_pn.add_hamiltonians(pn_n)

    total_matrix = total_pn.to_sparse_matrix()
    value = np.vdot(ground_vec, total_matrix @ ground_vec)

    return value

def gauss_law_values_sparse(ground_vec, lattice):
    gauss_law_values_dict = {}

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            n = lattice.reverse_labels[(x, y)]
            g_n = gauss_operator_n(lattice, n)

            g_n_matrix = g_n.to_sparse_matrix()
            value = np.vdot(ground_vec, g_n_matrix @ ground_vec)

            gauss_law_values_dict[(x, y)] = value

    return gauss_law_values_dict

def total_gauss_law_sparse(ground_vec, lattice):
    hamiltonian = gauss_hamiltonian_linear(lattice)
    matrix = hamiltonian.to_sparse_matrix()
    value = np.vdot(ground_vec, matrix @ ground_vec)
    return value

def observe_and_print_sparse(ground_vec, lattice):
    electric_field_values_dict = electric_field_values_sparse(ground_vec, lattice)
    magnetic_field_values_dict = magnetic_field_values_sparse(ground_vec, lattice)
    particle_number_value_dict = particle_number_values_sparse(ground_vec, lattice)
    gauss_law_values_dict = gauss_law_values_sparse(ground_vec, lattice)

    # _total_particle_number = total_particle_number_sparse(ground_vec, lattice)
    _total_particle_number = sum(particle_number_value_dict.values())
    _total_charge = total_charge_sparse(ground_vec, lattice)
    _total_gauss_law = total_gauss_law_sparse(ground_vec, lattice)

    electric_field_values_dict = smart_round(electric_field_values_dict, 6)
    magnetic_field_values_dict = smart_round(magnetic_field_values_dict, 6)
    particle_number_value_dict = smart_round(particle_number_value_dict, 6)
    gauss_law_values_dict = smart_round(gauss_law_values_dict, 6)

    _total_particle_number = smart_round(_total_particle_number, 6)
    _total_charge = smart_round(_total_charge, 6)
    _total_gauss_law = smart_round(_total_gauss_law, 6)

    print("electric_field_values_dict:", electric_field_values_dict)
    print("magnetic_field_values_dict:", magnetic_field_values_dict)
    print("particle_number_value_dict:", particle_number_value_dict)
    print("gauss_law_values_dict (no charges):", gauss_law_values_dict)
    print("total_charge:", _total_charge)
    print("total_gauss (with charges):", _total_gauss_law)
    print("total_particle_number:", _total_particle_number)
    
    return _total_particle_number
    
def electric_field_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    electric_field_values_dict = {}
    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            electric_hamiltonian = Hamiltonian(lattice.n_qubits)
            electric_hamiltonian = electric_field_quadratic_term_n_direction(electric_hamiltonian, lattice, n, direction, lattice.dynamical_links_list)
            
            param_dict = dict(zip(thetas, thetas_values))
            circuit_values = circuit.assign_parameters(param_dict)
            value = measurer.expected_value_hamiltonian_qed(electric_hamiltonian, circuit_values, lattice, shots)
    
            electric_field_values_dict[(lattice.labels[n],direction)] = value
    return electric_field_values_dict

def magnetic_field_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    magnetic_field_values_dict = {}
    for n in lattice.plaquettes:
        magnetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        magnetic_hamiltonian = magnetic_term_n(magnetic_hamiltonian, lattice, n, lattice.dynamical_links_list)
        
        param_dict = dict(zip(thetas, thetas_values))
        circuit_values = circuit.assign_parameters(param_dict)
        value = measurer.expected_value_hamiltonian_qed(magnetic_hamiltonian, circuit_values, lattice, shots)
        
        magnetic_field_values_dict[lattice.labels[n]] = value
    return magnetic_field_values_dict

def total_charge_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    hamiltonian = charge_total_hamiltonian(lattice)
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    value = measurer.expected_value_hamiltonian_qed(hamiltonian, circuit_values, lattice, shots)
    return value

def particle_number_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    particle_number_values_dict = {}
    total_pn = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            param_dict = dict(zip(thetas, thetas_values))
            circuit_values = circuit.assign_parameters(param_dict)
            value = measurer.expected_value_hamiltonian_qed(pn_n, circuit_values, lattice, shots)
            particle_number_values_dict[(x,y)] = value

    return particle_number_values_dict

def total_particle_number_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    total_pn = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            total_pn.add_hamiltonians(pn_n)
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    value = measurer.expected_value_hamiltonian_qed(total_pn, circuit_values, lattice, shots)
    return value

def gauss_law_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    gauss_law_values_dict = {}
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            n = lattice.reverse_labels[(x, y)]
            g_n = gauss_operator_n(lattice, n)
            
            param_dict = dict(zip(thetas, thetas_values))
            circuit_values = circuit.assign_parameters(param_dict)
            value = measurer.expected_value_hamiltonian_qed(g_n, circuit_values, lattice, shots)
            gauss_law_values_dict[(x,y)] = value

    return gauss_law_values_dict

def total_gauss_law_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    hamiltonian = gauss_hamiltonian_linear(lattice)
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    value = measurer.expected_value_hamiltonian_qed(hamiltonian, circuit_values, lattice, shots)
    return value
    
def observe_and_print_circuit(thetas, thetas_values, circuit, lattice, measurer, shots):
    electric_field_values_dict = electric_field_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots)
    magnetic_field_values_dict = magnetic_field_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots)
    particle_number_value_dict = particle_number_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots)
    gauss_law_values_dict = gauss_law_values_circuit(thetas, thetas_values, circuit, lattice, measurer, shots)

    # _total_particle_number = total_particle_number_sparse(ground_vec, lattice)
    _total_particle_number = sum(particle_number_value_dict.values())
    _total_charge = total_charge_circuit(thetas, thetas_values, circuit, lattice, measurer, shots)
    _total_gauss_law = total_gauss_law_circuit(thetas, thetas_values, circuit, lattice, measurer, shots)

    electric_field_values_dict = smart_round(electric_field_values_dict, 6)
    magnetic_field_values_dict = smart_round(magnetic_field_values_dict, 6)
    particle_number_value_dict = smart_round(particle_number_value_dict, 6)
    gauss_law_values_dict = smart_round(gauss_law_values_dict, 6)

    _total_particle_number = smart_round(_total_particle_number, 6)
    _total_charge = smart_round(_total_charge, 6)
    _total_gauss_law = smart_round(_total_gauss_law, 6)

    print("electric_field_values_dict:", electric_field_values_dict)
    print("magnetic_field_values_dict:", magnetic_field_values_dict)
    print("particle_number_value_dict:", particle_number_value_dict)
    print("gauss_law_values_dict (no charges):", gauss_law_values_dict)
    print("total_charge:", _total_charge)
    print("total_gauss (with charges):", _total_gauss_law)
    print("total_particle_number:", _total_particle_number)
    
    return _total_particle_number
    
def _statevector_from_params(thetas, thetas_values, circuit):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    circuit_nom = circuit_values.remove_final_measurements(inplace=False)
    return qiskit.quantum_info.Statevector.from_instruction(circuit_nom)
    
def electric_field_values_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    values = {}

    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            H = Hamiltonian(lattice.n_qubits)
            H = electric_field_quadratic_term_n_direction(
                H, lattice, n, direction, lattice.dynamical_links_list
            )
            values[(lattice.labels[n], direction)] = np.real(
                psi.expectation_value(H.to_qiskit())
            )

    return values

def magnetic_field_values_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    values = {}

    for n in lattice.plaquettes:
        H = Hamiltonian(lattice.n_qubits)
        H = magnetic_term_n(H, lattice, n, lattice.dynamical_links_list)
        values[lattice.labels[n]] = np.real(
            psi.expectation_value(H.to_qiskit())
        )

    return values

def total_charge_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    H = charge_total_hamiltonian(lattice)
    return np.real(psi.expectation_value(H.to_qiskit()))

def particle_number_values_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    values = {}

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            H = particle_n_hamiltonian(lattice, (x, y))
            values[(x, y)] = np.real(
                psi.expectation_value(H.to_qiskit())
            )

    return values

def total_particle_number_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    H = Hamiltonian(lattice.n_qubits)

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            H.add_hamiltonians(particle_n_hamiltonian(lattice, (x, y)))

    return np.real(psi.expectation_value(H.to_qiskit()))

def gauss_law_values_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    values = {}

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            n = lattice.reverse_labels[(x, y)]
            H = gauss_operator_n(lattice, n)
            values[(x, y)] = np.real(
                psi.expectation_value(H.to_qiskit())
            )

    return values

def total_gauss_law_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    H = gauss_hamiltonian_linear(lattice)
    return np.real(psi.expectation_value(H.to_qiskit()))

def observe_and_print_noiseless(thetas, thetas_values, circuit, lattice):
    ef = electric_field_values_noiseless(thetas, thetas_values, circuit, lattice)
    mf = magnetic_field_values_noiseless(thetas, thetas_values, circuit, lattice)
    pn = particle_number_values_noiseless(thetas, thetas_values, circuit, lattice)
    gl = gauss_law_values_noiseless(thetas, thetas_values, circuit, lattice)

    total_pn = sum(pn.values())
    total_charge = total_charge_noiseless(thetas, thetas_values, circuit, lattice)
    total_gauss = total_gauss_law_noiseless(thetas, thetas_values, circuit, lattice)

    ef = smart_round(ef, 6)
    mf = smart_round(mf, 6)
    pn = smart_round(pn, 6)
    gl = smart_round(gl, 6)

    total_pn = smart_round(total_pn, 6)
    total_charge = smart_round(total_charge, 6)
    total_gauss = smart_round(total_gauss, 6)

    print("electric_field_values_dict:", ef)
    print("magnetic_field_values_dict:", mf)
    print("particle_number_value_dict:", pn)
    print("gauss_law_values_dict (no charges):", gl)
    print("total_charge:", total_charge)
    print("total_gauss (with charges):", total_gauss)
    print("total_particle_number:", total_pn)

    return total_pn

def initiate_circuit_observables(parameters, lattice):
    measurer = Measurements_gpu(parameters['simulator'])
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_gauge_qubits)
    
    builder.initialize_fermions(lattice)
    
    n_slice = builder.iSwap_block_calculate_qed(lattice, parameters['n_fermion_layers'])
    
    thetas_per_gauge = {2: 2, 
                        3: 4}
                        
    n_gauge_thetas = thetas_per_gauge[lattice.qubits_per_gauge] * lattice.n_dynamical_links
    n_fermion_thetas = n_slice * parameters['n_fermion_layers']
    n_extra_thetas = lattice.n_gauge_qubits * parameters['n_extra_layers']
    
    total_thetas = n_fermion_thetas + n_gauge_thetas + n_extra_thetas
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_fermion_thetas]
    gauge_thetas = thetas[n_fermion_thetas:n_fermion_thetas + n_gauge_thetas]
    extra_thetas = thetas[n_fermion_thetas + n_gauge_thetas:]
    
    for j in range(parameters['n_fermion_layers']):
        builder.iSwap_block_qed(fermion_thetas[n_slice*j:n_slice*(j+1)], lattice)
        
    builder.gauge_block(gauge_thetas, parameters['gauge_truncation'])
    
    for j in range(parameters['n_extra_layers']):
        builder.R_Y_layer_gauss(extra_thetas[lattice.n_gauge_qubits*j:lattice.n_gauge_qubits*(j+1)])
        
    circuit = builder.build()
    # print(circuit.draw())
    
    return circuit, thetas, total_thetas, n_fermion_thetas
    
def build_and_draw(parameters, lattice):
    measurer = Measurements_gpu(parameters['simulator'])
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_gauge_qubits)
    
    builder.initialize_fermions(lattice)
    
    n_slice = builder.iSwap_block_calculate_qed(lattice, parameters['n_fermion_layers'])
    
    thetas_per_gauge = {2: 2, 
                        3: 4}
                        
    n_gauge_thetas = thetas_per_gauge[lattice.qubits_per_gauge] * lattice.n_dynamical_links
    n_fermion_thetas = n_slice * parameters['n_fermion_layers']
    n_extra_thetas = lattice.n_gauge_qubits * parameters['n_extra_layers']
    
    total_thetas = n_fermion_thetas + n_gauge_thetas + n_extra_thetas
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_fermion_thetas]
    gauge_thetas = thetas[n_fermion_thetas:n_fermion_thetas + n_gauge_thetas]
    extra_thetas = thetas[n_fermion_thetas + n_gauge_thetas:]
    
    for j in range(parameters['n_fermion_layers']):
        builder.iSwap_block_qed(fermion_thetas[n_slice*j:n_slice*(j+1)], lattice)
        
    builder.gauge_block(gauge_thetas, parameters['gauge_truncation'])
    
    for j in range(parameters['n_extra_layers']):
        builder.R_Y_layer_gauss(extra_thetas[lattice.n_gauge_qubits*j:lattice.n_gauge_qubits*(j+1)])
        
    circuit = builder.build()
    print(circuit.draw())
    
    return circuit


def qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)
    ev = measurer.expected_value_hamiltonian_qed(hamiltonian, circuit_values, lattice, shots)
    return ev
    
def qed_vqe_noiseless(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
    param_dict = dict(zip(thetas, thetas_values))
    circuit_values = circuit.assign_parameters(param_dict)

    # circuit_nom = circuit_values.remove_final_measurements(inplace=False)
    psi = qiskit.quantum_info.Statevector.from_instruction(circuit_values)

    H_qiskit = hamiltonian.to_qiskit()

    ev = np.real(psi.expectation_value(H_qiskit))
    return ev
    
def qed_vqe_noiseless_vectorized(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, shots):
    N = len(thetas)
    param_dict = dict(zip(thetas, thetas_values))
    psi0 = qiskit.quantum_info.Statevector.from_instruction(circuit.assign_parameters(param_dict))
    
    H_qiskit = hamiltonian.to_qiskit()
    energy = np.real(psi0.expectation_value(H_qiskit))
    
    # Parameter-shift gradient
    shifts = np.pi / 2
    grads = np.zeros(N)
    
    # Shift all parameters "plus" and "minus" in a batched way
    for k in range(N):
        thetas_plus  = thetas_values.copy()
        thetas_minus = thetas_values.copy()
        thetas_plus[k]  += shifts
        thetas_minus[k] -= shifts
        
        psi_plus  = qiskit.quantum_info.Statevector.from_instruction(circuit.assign_parameters(dict(zip(thetas, thetas_plus))))
        psi_minus = qiskit.quantum_info.Statevector.from_instruction(circuit.assign_parameters(dict(zip(thetas, thetas_minus))))
        
        grads[k] = 0.5 * (np.real(psi_plus.expectation_value(H_qiskit)) - np.real(psi_minus.expectation_value(H_qiskit)))
    
    return energy, grads

def full_runner(n_fermion_thetas, n_gauge_thetas, thetas, circuit, hamiltonian, lattice, measurer, parameters, sparse_test = False, noisy = False, diagnostics = False, include_ps = False):
    if diagnostics: print("Runner called")
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

    def full_function(thetas_values):
        if noisy:
            cost = qed_vqe(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
        else:
            cost = qed_vqe_noiseless(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
            
        if diagnostics:
            print('-'*10)
            print("Energy:",cost)
            if noisy:
                observe_and_print_circuit(thetas, thetas_values, circuit, lattice, measurer, parameters['shots'])
            else:
                observe_and_print_noiseless(thetas, thetas_values, circuit, lattice)
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
        
    guess = np.random.uniform(-0.08, 0.08, size = len(thetas))
    mini = scipy.optimize.minimize(
        fun=full_function,
        x0=guess,
        method="COBYLA",
        options={"maxiter": 2000, "tol": 1e-6}
    )
    final_thetas = mini.x
    final_value = full_function(final_thetas)
    
    if include_ps:
        mini = scipy.optimize.minimize(
            fun=full_function,
            jac=grad_function,
            x0=final_thetas,
            method="L-BFGS-B",
            options={"maxiter": 50}
        )
        final_final_thetas = mini.x
        final_quantum_value = full_function(mini.x)
    else:
        final_final_thetas = final_thetas
        final_quantum_value = final_value
    
    return final_final_thetas, final_quantum_value

def fuller_runner(parameters, lattice, extra_parameters):
    if extra_parameters['diagnostics']: print("System initializing")
    circuit, thetas, total_thetas, n_fermion_thetas = initiate_circuit_observables(parameters, lattice)
    n_gauge_thetas = total_thetas - n_fermion_thetas
    
    measurer = Measurements_gpu(parameters['simulator'], lattice.n_qubits)
    
    hamiltonian = generate_qed_hamiltonian(parameters, lattice)
    
    thetas_values, energy = full_runner(n_fermion_thetas, n_gauge_thetas, thetas, circuit, hamiltonian, lattice, measurer, parameters, extra_parameters['sparse_test'], extra_parameters['noisy'], extra_parameters['diagnostics'], extra_parameters['include_ps'])
    
    # thetas_values, energy = p_s_runner_fast(thetas, circuit, hamiltonian, lattice, measurer, parameters, diagnostics=False)
    
    cost = qed_vqe_noiseless(thetas, thetas_values, circuit, hamiltonian, lattice, measurer, parameters['shots'])
    observe_and_print_noiseless(thetas, thetas_values, circuit, lattice)
    print("Energy:",cost)
    
    return thetas_values, energy