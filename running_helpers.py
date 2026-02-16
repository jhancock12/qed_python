from modules import *
from classes import *
from circuit_helpers import *
from plot_helpers import *
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