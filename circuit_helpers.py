from modules import *
from classes import *
from QED_hamiltonian import *

def initiate_circuit_observables(parameters, lattice):
    measurer = Measurements_gpu(parameters['simulator'])
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_gauge_qubits)
    
    builder.initialize_fermions(lattice)
    # builder.initialize_gauge(lattice)
    
    n_slice = builder.iSwap_block_calculate_qed(lattice, parameters['n_fermion_layers'])
    
    thetas_per_gauge = {2: 2, 
                        3: 4}
                        
    n_gauge_thetas = thetas_per_gauge[lattice.qubits_per_gauge] * lattice.n_dynamical_links
    n_fermion_thetas = n_slice * parameters['n_fermion_layers']
    n_extra_thetas = lattice.n_fermion_qubits
    
    total_thetas = n_fermion_thetas + n_gauge_thetas + n_extra_thetas
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_fermion_thetas]
    gauge_thetas = thetas[n_fermion_thetas:n_fermion_thetas + n_gauge_thetas]
    extra_thetas = thetas[n_fermion_thetas + n_gauge_thetas:]
    
    for j in range(parameters['n_fermion_layers']):
        builder.iSwap_block_qed(fermion_thetas[n_slice*j:n_slice*(j+1)], lattice)
        
    builder.gauge_block(gauge_thetas, parameters['gauge_truncation'])
    
    for j in range(lattice.n_fermion_qubits):
        builder.circuit.rz(extra_thetas[j], lattice.n_gauge_qubits + j)
        
    circuit = builder.build()
    # print(circuit.draw())
    
    return circuit, thetas, total_thetas, n_fermion_thetas

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

def initiate_circuit_observables_match_paper(parameters, lattice):
    measurer = Measurements_gpu(parameters['simulator'])
    builder = CircuitBuilder(lattice.n_fermion_qubits, lattice.n_dynamical_gauge_qubits)
    
    builder.initialize_fermions(lattice)

    builder.initialize_gauge(lattice)
    
    n_slice = builder.iSwap_block_calculate_qed(lattice, parameters['n_fermion_layers'])
    
    thetas_per_gauge = {2: 2, 
                        3: 4}
    
    n_gauge_thetas = thetas_per_gauge[lattice.qubits_per_gauge] * lattice.n_dynamical_links + 4
    n_fermion_thetas = n_slice * parameters['n_fermion_layers'] + lattice.qubits_per_gauge * lattice.n_dynamical_links
    n_extra_thetas = lattice.n_fermion_qubits

    def controlled_iSwap(circuit, theta, ctrl, j, k):
        sub = qiskit.QuantumCircuit(2)
        sub.ryy((theta/2), 0, 1)
        sub.rxx((theta/2), 0, 1)

        gate = sub.to_gate(label="iSwap")
        cgate = gate.control(1)

        circuit.append(cgate, [ctrl, j, k])
    
    total_thetas = n_fermion_thetas + n_gauge_thetas + n_extra_thetas
    thetas = qiskit.circuit.ParameterVector('θ', total_thetas)
    fermion_thetas = thetas[:n_fermion_thetas]
    gauge_thetas = thetas[n_fermion_thetas:n_fermion_thetas + n_gauge_thetas]
    extra_thetas = thetas[n_fermion_thetas + n_gauge_thetas:]
    
    for j in range(parameters['n_fermion_layers']):
        builder.iSwap_block_qed(fermion_thetas[n_slice*j:n_slice*(j+1)], lattice)
    labels0 = [lattice.dynamical_link_indexing[lattice.dynamical_links_list[0]] * lattice.qubits_per_gauge + l for l in range(lattice.qubits_per_gauge)]
    labels1 = [lattice.dynamical_link_indexing[lattice.dynamical_links_list[1]] * lattice.qubits_per_gauge + l for l in range(lattice.qubits_per_gauge)]

    # Hardcoded atm, works for two links
    builder.circuit.ry(gauge_thetas[0], labels0[0])
    builder.circuit.ry(gauge_thetas[2], labels1[0])
    builder.circuit.cry(gauge_thetas[1], labels0[0], labels0[1])
    builder.circuit.cry(gauge_thetas[3], labels0[0], labels1[0])
    builder.circuit.cry(gauge_thetas[4], labels0[1], labels1[0])
    builder.circuit.cry(gauge_thetas[5], labels1[0], labels1[1])
    builder.circuit.mcry(gauge_thetas[6], [labels0[0], labels1[0]], labels1[1])
    builder.circuit.mcry(gauge_thetas[7], [labels0[1], labels1[0]], labels1[1])
    
    theta_counter = 0
    for j in range(len(lattice.dynamical_links_list)):
        start = lattice.dynamical_links_list[j][0]
        direction = lattice.dynamical_links_list[j][1]
        end = list(copy.copy(start))
        end[direction - 1] += 1
        end = tuple(end)
        labels = [lattice.dynamical_link_indexing[lattice.dynamical_links_list[j]] * lattice.qubits_per_gauge + l for l in range(lattice.qubits_per_gauge)]
        for k in range(lattice.qubits_per_gauge):
            controlled_iSwap(builder.circuit, fermion_thetas[n_slice * parameters['n_fermion_layers'] + theta_counter], labels[k], lattice.reverse_labels[start] + lattice.n_gauge_qubits, lattice.reverse_labels[end] + lattice.n_gauge_qubits)
            theta_counter += 1

    for j in range(lattice.n_fermion_qubits):
        builder.circuit.rz(extra_thetas[j], lattice.n_gauge_qubits + j)
        
    circuit = builder.build()
    
    return circuit, thetas, total_thetas, n_fermion_thetas

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
            H = electric_field_linear_term_n_direction(
                H, lattice, n, direction, lattice.dynamical_links_list
            )
            values[(lattice.labels[n], direction)] = np.real(
                psi.expectation_value(H.to_qiskit())
            )

    return values

def total_electric_field_values_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    values = {}

    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            H = Hamiltonian(lattice.n_qubits)
            H = electric_field_linear_term_n_direction(
                H, lattice, n, direction, lattice.dynamical_links_list
            )
            H.add_term('I'*lattice.n_qubits, lattice.background_electric_field.get((lattice.labels[n], direction), 0.0))
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

def charge_values_noiseless(thetas, thetas_values, circuit, lattice):
    psi = _statevector_from_params(thetas, thetas_values, circuit)
    values = {}

    for n in range(lattice.n_fermion_qubits):
        indices = lattice.labels[n]
        H = Hamiltonian(lattice.n_qubits)
        H = charge_n_hamiltonian(lattice, indices)
        values[indices] = np.real(
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
    t_ef = smart_round(t_ef, 6)
    mf = smart_round(mf, 6)
    pn = smart_round(pn, 6)
    gl = smart_round(gl, 6)

    total_pn = smart_round(total_pn, 6)
    total_charge = smart_round(total_charge, 6)
    total_gauss = smart_round(total_gauss, 6)
    for link in lattice.gauss_law_background:
        print("bg link:", link, "value:", lattice.gauss_law_background[link])
    print("electric_field_values_dict:", ef)
    print("total_electric_field_values_dict (includes background):", t_ef)
    print("magnetic_field_values_dict:", mf)
    print("particle_number_value_dict:", pn)
    print("gauss_law_values_dict:", gl)
    print("total_charge:", total_charge)
    print("total_gauss:", total_gauss)
    print("total_particle_number:", total_pn)
    results_dict = {
        'electric_field_dict' : ef,
        'magnetic_field_dict' : mf,
        'particle_number_dict' : pn,
        'gauss_law_dict' : gl,
        'total_charge' : total_charge,
        'gauss_law_total' : total_gauss,
        'particle_number_total' : total_pn
        }
    

    return results_dict

def observe_and_print_noiseless_reduced(thetas, thetas_values, circuit, lattice, gauss_equations):
    ef = electric_field_values_noiseless(thetas, thetas_values, circuit, lattice) 
    mf = magnetic_field_values_noiseless(thetas, thetas_values, circuit, lattice)
    pn = particle_number_values_noiseless(thetas, thetas_values, circuit, lattice) 
    gl = gauss_law_values_noiseless(thetas, thetas_values, circuit, lattice) 
    
    dynamical_links_list = lattice.dynamical_links_list
    c = charge_values_noiseless(thetas, thetas_values, circuit, lattice)
    total_pn = sum(pn.values())
    total_charge = total_charge_noiseless(thetas, thetas_values, circuit, lattice)
    total_gauss = total_gauss_law_noiseless(thetas, thetas_values, circuit, lattice)
    
    
    for dependant_variable in gauss_equations['dependant_variables']:
        equation = gauss_equations['equations'][dependant_variable]
        sub_ins = {}
        for var in gauss_equations['independant_variables']:
            if var in list(gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[gauss_equations['reverse_link_variable_dict'][var]]
            else:
                n_charge = int(str(var)[1:])
                sub_ins[var] = c[lattice.labels[n_charge]]
        ef[gauss_equations['reverse_link_variable_dict'][dependant_variable]] = complex(equation.subs(sub_ins))
     
    n = 0   
    for equation in gauss_equations['equations']:
        sub_ins = {}
        for var in gauss_equations['independant_variables'] + gauss_equations['dependant_variables']:
            if var in list(gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[gauss_equations['reverse_link_variable_dict'][var]]
            else:
                n_charge = int(str(var)[1:])
                sub_ins[var] = c[lattice.labels[n_charge]]
        gl[lattice.labels[n]] = complex(equation.subs(sub_ins))
        n += 1
    

    ef = smart_round(ef, 6)
    mf = smart_round(mf, 6)
    pn = smart_round(pn, 6)
    gl = smart_round(gl, 6)

    total_pn = smart_round(total_pn, 6)
    total_charge = smart_round(total_charge, 6)
    total_gauss = smart_round(total_gauss, 6)
    for link in lattice.gauss_law_background:
        print("bg link:", link, "value:", lattice.gauss_law_background[link])
    print("electric_field_values_dict:", ef)
    print("charge_field_values_dict:", c)
    print("magnetic_field_values_dict:", mf)
    print("particle_number_value_dict:", pn)
    print("gauss_law_values_dict:", gl)
    print("total_charge:", total_charge)
    print("total_gauss:", total_gauss)
    print("total_particle_number:", total_pn)
    results_dict = {
        'electric_field_dict' : ef,
        'magnetic_field_dict' : mf,
        'particle_number_dict' : pn,
        'gauss_law_dict' : gl,
        'total_charge' : total_charge,
        'gauss_law_total' : total_gauss,
        'particle_number_total' : total_pn
        }
    # print("total_electric_field_values_dict (includes background):", t_ef)

    return results_dict