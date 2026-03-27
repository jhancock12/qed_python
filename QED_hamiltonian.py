from classes import *
    
def electric_field_term_n_direction(hamiltonian, lattice, n, direction):
    # print("Linear EF called")
    index = lattice.labels[n]
    if (index, direction) in lattice.dynamical_links_list:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * ((link_index) * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)

        for key in list(lattice.E_dict):
            term = gauge_before + key + gauge_after + fermion_string
            coeff = lattice.E_dict[key]
            hamiltonian.add_term(term, coeff)
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 0)

    return hamiltonian

def electric_field_term_n(hamiltonian, lattice, n, dynamical_links):
    for direction in lattice.directions[n]:
        E_temp = Hamiltonian(lattice.n_qubits)
        E_temp = electric_field_term_n_direction(E_temp, lattice, n, direction, dynamical_links)
        hamiltonian.add_hamiltonians(E_temp)
    return hamiltonian

def background_electric_field_term_n(hamiltonian, lattice, n, dynamical_links):
    E_temp = Hamiltonian(lattice.n_qubits)
    for direction in lattice.directions[n]:
        E_temp = electric_field_linear_term_n_direction(E_temp, lattice, n, direction, dynamical_links)
        hamiltonian.add_hamiltonians(E_temp)
    return hamiltonian
         
def U_term_n(hamiltonian, lattice, n, direction, dynamical_links): 
    index = lattice.labels[n]
    if (index,direction) in dynamical_links:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * (link_index * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)
        
        for key in list(lattice.U_dict):
            term = gauge_before + key + gauge_after + fermion_string
            value = lattice.U_dict[key]
            hamiltonian.add_term(term, value)       
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 1.0)
    return hamiltonian

def magnetic_term_n(hamiltonian, lattice, n, dynamical_links):
    index = lattice.labels[n]
    ns_directions = [
        (lattice.reverse_labels[(index[0],index[1])], 1),
        (lattice.reverse_labels[(index[0]+1,index[1])], 2),
        (lattice.reverse_labels[(index[0],index[1]+1)], 1),
        (lattice.reverse_labels[(index[0],index[1])], 2)
    ]

    Us = [Hamiltonian(lattice.n_qubits) for _ in range(4)]
    for i in range(4): Us[i] = U_term_n(Us[i], lattice, ns_directions[i][0], ns_directions[i][1], dynamical_links)
    Us[2].hamiltonian = Us[2].to_conjugate()
    Us[3].hamiltonian = Us[3].to_conjugate()

    P_n = Hamiltonian(lattice.n_qubits)
    P_n.add_term('I'*lattice.n_qubits, 1.0)

    for i in range(4): P_n.multiply_hamiltonians(Us[i])

    P_n.to_conjugate()

    P_n_dagger = Hamiltonian(lattice.n_qubits)
    P_n_dagger.hamiltonian = P_n.conjugate

    hamiltonian.add_hamiltonians(P_n)
    hamiltonian.add_hamiltonians(P_n_dagger)

    return hamiltonian

def creation_operator_n(hamiltonian, lattice, n):
    gauge_string = 'I' * lattice.n_dynamical_gauge_qubits
    fermions_before = 'Z' * n
    fermions_after = 'I' * (lattice.n_fermion_qubits - (n + 1))
    coeff = (1j)**n

    hamiltonian.add_term(gauge_string + fermions_before + 'X' + fermions_after, coeff / 2)
    hamiltonian.add_term(gauge_string + fermions_before + 'Y' + fermions_after, -1j * coeff / 2)
    return hamiltonian

def annihilation_operator_n(hamiltonian, lattice, n):
    gauge_string = 'I' * lattice.n_dynamical_gauge_qubits
    fermions_before = 'Z' * n
    fermions_after = 'I' * (lattice.n_fermion_qubits - (n + 1))
    coeff = (-1j)**n

    hamiltonian.add_term(gauge_string + fermions_before + 'X' + fermions_after, coeff / 2)
    hamiltonian.add_term(gauge_string + fermions_before + 'Y' + fermions_after, 1j * coeff / 2)
    return hamiltonian
    
def mass_term_n(hamiltonian, lattice, n):
    # print("Mass called")
    operator = Hamiltonian(lattice.n_qubits)
    annihilation_operator = Hamiltonian(lattice.n_qubits)
    
    operator = creation_operator_n(operator, lattice, n) # M -> c
    annihilation_operator = annihilation_operator_n(annihilation_operator, lattice, n)
    operator.multiply_hamiltonians(annihilation_operator) # M -> ca
    
    indices = lattice.labels[n]
    coeff = ((-1)**(indices[0] + indices[1]))
    operator.hamiltonian = multiply_hamiltonian_by_constant(operator.hamiltonian, coeff)
    
    hamiltonian.add_hamiltonians(operator)
    return hamiltonian

def kinetic_subterm_n(lattice, n, direction, dynamical_links):
    indices = lattice.labels[n]

    if direction == 1:
        indices_mu = (indices[0]+1, indices[1])
    elif direction == 2:
        indices_mu = (indices[0], indices[1]+1)

    n_mu = lattice.reverse_labels[indices_mu]

    hamiltonian = Hamiltonian(lattice.n_qubits)
    hamiltonian.add_term('I'*lattice.n_qubits, 1.0)

    creation_hamiltonian = Hamiltonian(lattice.n_qubits)
    U_hamiltonian = Hamiltonian(lattice.n_qubits)
    annihilation_hamiltonian = Hamiltonian(lattice.n_qubits)

    creation_hamiltonian = creation_operator_n(creation_hamiltonian, lattice, n)
    U_hamiltonian = U_term_n(U_hamiltonian, lattice, n, direction, dynamical_links)
    U_hamiltonian.hamiltonian = U_hamiltonian.to_conjugate()
    annihilation_hamiltonian = annihilation_operator_n(annihilation_hamiltonian, lattice, n_mu)

    hamiltonian.multiply_hamiltonians(creation_hamiltonian) # M -> c
    hamiltonian.multiply_hamiltonians(U_hamiltonian) # M -> cU
    hamiltonian.multiply_hamiltonians(annihilation_hamiltonian) # M -> cUa
    
    return hamiltonian

def kinetic_term_n(hamiltonian, lattice, n, dynamical_links):
    temp_hamiltonian = Hamiltonian(lattice.n_qubits)
    temp_hamiltonian_dagger = Hamiltonian(lattice.n_qubits)
    indices = lattice.labels[n]
    coeffs = {
        1: 1j,
        2: -1*((-1)**(indices[0] + indices[1]))
    }

    for direction in lattice.directions[n]:
        temp_hamiltonian = kinetic_subterm_n(lattice, n, direction, dynamical_links)
        temp_hamiltonian.to_conjugate()
        temp_hamiltonian_dagger.hamiltonian = temp_hamiltonian.conjugate

        if direction == 1: temp_hamiltonian_dagger.hamiltonian = multiply_hamiltonian_by_constant(temp_hamiltonian_dagger.hamiltonian, -1)
        temp_hamiltonian.add_hamiltonians(temp_hamiltonian_dagger)
        
        temp_hamiltonian.hamiltonian = multiply_hamiltonian_by_constant(temp_hamiltonian.hamiltonian, coeffs[direction])
        hamiltonian.add_hamiltonians(temp_hamiltonian)
    return hamiltonian
 
def particle_n_hamiltonian(lattice, indices):
    x, y = indices
    pn_hamiltonian = Hamiltonian(lattice.n_qubits)
    I_term = 'I' * lattice.n_qubits

    # site index in qubit ordering
    n = lattice.get_index(x, y)
    
    # Identity term with parity factor
    sign = (-1)**(x + y)
    pn_hamiltonian.add_term(I_term, 0.5)

    # Z term acting on fermion qubit
    Z_term = list(I_term)
    Z_term[lattice.n_dynamical_gauge_qubits + n] = 'Z'
    pn_hamiltonian.add_term(''.join(Z_term), -0.5*sign)
    
    return pn_hamiltonian

def particle_number_hamiltonian(lattice):
    total_pn = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            pn_n = particle_n_hamiltonian(lattice, (x, y))
            total_pn.add_hamiltonians(pn_n)
    return total_pn

def charge_n_hamiltonian(lattice, indices):
    x, y = indices
    n = lattice.get_index(x, y)
    charge_hamiltonian = Hamiltonian(lattice.n_qubits)

    I_term = 'I' * lattice.n_qubits
    Z_term = list(I_term)
    Z_term[lattice.n_dynamical_gauge_qubits + n] = 'Z'

    # staggered phase
    parity = (-1)**(x + y)

    charge_hamiltonian.add_term(''.join(Z_term), -0.5)
    charge_hamiltonian.add_term(I_term, parity * 0.5)

    return charge_hamiltonian

def charge_total_hamiltonian(lattice):
    total_charge = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            ch_n = charge_n_hamiltonian(lattice, (x, y))
            total_charge.add_hamiltonians(ch_n)
    return total_charge
    
def electric_solve_gauss(lattice):
    gauss_equations = lattice.gauss_equations

    link_variables = gauss_equations['link_variables']
    charge_variables = gauss_equations['charge_variables']
    link_variable_dict = gauss_equations['link_variable_dict']
    reverse_link_variable_dict = gauss_equations['reverse_link_variable_dict']
        
    eqs = gauss_equations['equations']
    dependant_variables = gauss_equations['dependant_variables']
    independant_variables = gauss_equations['independant_variables']
    
    solution = gauss_equations['solution']
    
    hamiltonian_variables = {}
    for variable in link_variables:
        temp_hamiltonian = Hamiltonian(lattice.n_qubits)
        link = reverse_link_variable_dict[variable]
        hamiltonian_variables[variable] = electric_field_term_n_direction(temp_hamiltonian, lattice, lattice.reverse_labels[link[0]], link[1])
    
    for n in range(lattice.n_fermion_qubits):
        variable = charge_variables[n]
        hamiltonian_variables[variable] = charge_n_hamiltonian(lattice, lattice.labels[n])
        
    total_hamiltonian = Hamiltonian(lattice.n_qubits)
    for dependant_variable in dependant_variables:
        direction = reverse_link_variable_dict[dependant_variable][1]
        if dependant_variable in solution and dependant_variable not in charge_variables:
            equation = solution[dependant_variable]
            temp_hamiltonian = Hamiltonian(lattice.n_qubits)
            for variable in independant_variables:
                coefficient = complex(equation.coeff(variable))
                temp_hamiltonian_inside = copy.copy(hamiltonian_variables[variable])
                
                temp_hamiltonian_inside.hamiltonian = multiply_hamiltonian_by_constant(temp_hamiltonian_inside.hamiltonian, coefficient)
                
                temp_hamiltonian.add_hamiltonians(temp_hamiltonian_inside)
            
            constant = complex(equation.subs({v: 0 for v in independant_variables}))
            constant_hamiltonian = Hamiltonian(lattice.n_qubits)
            constant_hamiltonian.add_term("I"*lattice.n_qubits, constant + lattice.E_0[direction - 1])
            temp_hamiltonian.add_hamiltonians(constant_hamiltonian)
            temp_hamiltonian.multiply_hamiltonians(temp_hamiltonian)
            temp_hamiltonian.cleanup()
            total_hamiltonian.add_hamiltonians(temp_hamiltonian)
    
    for variable in independant_variables:
        if variable not in charge_variables:
            direction = reverse_link_variable_dict[variable][1]
            temp_hamiltonian = copy.copy(hamiltonian_variables[variable])
            temp_hamiltonian.add_term("I"*lattice.n_qubits, lattice.E_0[direction - 1])
            temp_hamiltonian.multiply_hamiltonians(temp_hamiltonian)
            total_hamiltonian.add_hamiltonians(temp_hamiltonian)
    total_hamiltonian.cleanup()
    return total_hamiltonian  
   
def generate_qed_hamiltonian(parameters, lattice, to_print = False, mass_multi = 1, electric_multi = 1, magnetic_multi = 1, kinetic_multi = 1):
    full_hamiltonian = Hamiltonian(lattice.n_qubits)
    
    mass_coeff = parameters['m']
    electric_coeff = parameters['g']*parameters['g'] / 2
    magnetic_coeff = -1/(2*(parameters['a']*parameters['a'])*(parameters['g']*parameters['g']))
    kinetic_coeff = 1/(2*parameters['a'])

    # For testing
    mass_coeff *= mass_multi
    electric_coeff *= electric_multi
    magnetic_coeff *= magnetic_multi
    kinetic_coeff *= kinetic_multi

    mass_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    electric_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    magnetic_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    kinetic_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    BEF_hamiltonian_total = Hamiltonian(lattice.n_qubits)

    # Mass term
    for n in range(lattice.n_fermion_qubits):
        mass_hamiltonian = Hamiltonian(lattice.n_qubits)
        mass_hamiltonian = mass_term_n(mass_hamiltonian, lattice, n)
        mass_hamiltonian_total.add_hamiltonians(mass_hamiltonian)

    mass_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(mass_hamiltonian_total.hamiltonian, mass_coeff)
    mass_hamiltonian_total.cleanup()
    
    # Electric field term
    electric_hamiltonian_total = electric_solve_gauss(lattice)
    electric_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(electric_hamiltonian_total.hamiltonian, electric_coeff)
    
    # Magnetic field term
    for n in lattice.plaquettes:
        magnetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        magnetic_hamiltonian = magnetic_term_n(magnetic_hamiltonian, lattice, n, lattice.dynamical_links_list)
        magnetic_hamiltonian_total.add_hamiltonians(magnetic_hamiltonian)

    magnetic_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(magnetic_hamiltonian_total.hamiltonian, magnetic_coeff)
    magnetic_hamiltonian_total.cleanup()

    # Kinetic term
    for n in range(lattice.n_fermion_qubits):
        kinetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        kinetic_hamiltonian = kinetic_term_n(kinetic_hamiltonian, lattice, n, lattice.dynamical_links_list)
        kinetic_hamiltonian_total.add_hamiltonians(kinetic_hamiltonian)

    kinetic_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(kinetic_hamiltonian_total.hamiltonian, kinetic_coeff)
    kinetic_hamiltonian_total.cleanup()
    
    # Adding it all together
    full_hamiltonian.add_hamiltonians(mass_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(electric_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(magnetic_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(kinetic_hamiltonian_total)
    
    charge_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    for n in range(lattice.n_fermion_qubits):
        charge_hamiltonian = Hamiltonian(lattice.n_qubits)
        charge_hamiltonian = charge_n_hamiltonian(lattice, lattice.labels[n])
        charge_hamiltonian_total.add_hamiltonians(charge_hamiltonian)
    charge_hamiltonian_total.multiply_hamiltonians(charge_hamiltonian_total)
    charge_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(charge_hamiltonian_total.hamiltonian, parameters['charge_weight'])
    full_hamiltonian.add_hamiltonians(charge_hamiltonian_total)
    
    
    # Prints
    if to_print:
        print("Mass Hamiltonian:")
        print(mass_hamiltonian_total.hamiltonian)
        print("\n")
        print("Electric Hamiltonian:")
        print(electric_hamiltonian_total.hamiltonian)
        print("\n")
        print("Magnetic Hamiltonian:")
        print(magnetic_hamiltonian_total.hamiltonian)
        print("\n")
        print("Kinetic Hamiltonian")
        print(kinetic_hamiltonian_total.hamiltonian)
        print("\n")
        print("BEF Hamiltonian")
        print(BEF_hamiltonian_total.hamiltonian)
        print("\n")
            
    full_hamiltonian.cleanup()
    
    return full_hamiltonian