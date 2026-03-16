from classes import *
    
def electric_field_linear_term_n_direction(hamiltonian, lattice, n, direction, dynamical_links):
    # print("Linear EF called")
    index = lattice.labels[n]
    if (index, direction) in dynamical_links:
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
    
def electric_field_quadratic_term_n_direction(hamiltonian, lattice, n, direction, dynamical_links):
    index = lattice.labels[n]
    if (index, direction) in dynamical_links:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * ((link_index) * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)

        for key in list(lattice.E_dict):
            term = gauge_before + key + gauge_after + fermion_string
            coeff = lattice.E_dict[key]
            hamiltonian.add_term(term, coeff)
        hamiltonian.add_term('I'*lattice.n_qubits, lattice.E_0[direction - 1])
        hamiltonian.add_term('I'*lattice.n_qubits, lattice.gauss_law_background[(index, direction)])
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 0)
    hamiltonian.multiply_hamiltonians(hamiltonian)
    return hamiltonian

def electric_field_term_n(hamiltonian, lattice, n, dynamical_links):
    for direction in lattice.directions[n]:
        E_temp = Hamiltonian(lattice.n_qubits)
        E_temp = electric_field_quadratic_term_n_direction(E_temp, lattice, n, direction, dynamical_links)
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

    # bare staggered charge: -0.5 * parity * Z + 0.5 * I
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
    
def boundary_flux_at_site(lattice, indices):
    (x, y) = indices
    L_x, L_y = lattice.L_x, lattice.L_y
    [E_x, E_y] = lattice.E_0

    rho = 0.0

    if 1 in lattice.E_0_directions:
        if x == 0:
            rho -= E_x
        if x == L_x - 1:
            rho += E_x

    if 2 in lattice.E_0_directions:
        if y == 0:
            rho -= E_y
        if y == L_y - 1:
            rho += E_y

    return rho

def gauss_operator_n(lattice, n):
    G = Hamiltonian(lattice.n_qubits)
    indices = lattice.labels[n]
    for direction in [1,2]:
        if direction in lattice.directions[n]:
            G_on = Hamiltonian(lattice.n_qubits)
            G_on = electric_field_linear_term_n_direction(G_on, lattice, n, direction, lattice.dynamical_links_list)
            G_on.hamiltonian = multiply_hamiltonian_by_constant(G_on.hamiltonian, -1.0)
            G.add_hamiltonians(G_on)

        if direction == 1: new_indices = (indices[0] - 1, indices[1])
        elif direction == 2: new_indices = (indices[0], indices[1] - 1)

        if new_indices in list(lattice.reverse_labels):
            G_in = Hamiltonian(lattice.n_qubits)
            new_n = lattice.reverse_labels[new_indices]
            G_in = electric_field_linear_term_n_direction(G_in, lattice, new_n, direction, lattice.dynamical_links_list)
            G.add_hamiltonians(G_in)
            
    indices = lattice.labels[n]
    rho = boundary_flux_at_site(lattice, indices)
    G.add_term('I'*lattice.n_qubits, -rho)
    
    Q = charge_n_hamiltonian(lattice, indices)
    Q.hamiltonian = multiply_hamiltonian_by_constant(Q.hamiltonian, -1.0)
    G.add_hamiltonians(Q)
    G.cleanup()
    
    return G
    
def gauss_operator_n_full(lattice, n):
    G = Hamiltonian(lattice.n_qubits)
    indices = lattice.labels[n]
    x, y = indices

    # dynamical electric field part
    for direction in [1, 2]:
        # outgoing
        if direction in lattice.directions[n]:
            G_on = Hamiltonian(lattice.n_qubits)
            G_on = electric_field_linear_term_n_direction(
                G_on, lattice, n, direction, lattice.dynamical_links_list
            )
            G_on.hamiltonian = multiply_hamiltonian_by_constant(G_on.hamiltonian, -1.0)
            G.add_hamiltonians(G_on)

        # incoming
        if direction == 1:
            new_indices = (x - 1, y)
        else:
            new_indices = (x, y - 1)

        if new_indices in lattice.reverse_labels:
            G_in = Hamiltonian(lattice.n_qubits)
            new_n = lattice.reverse_labels[new_indices]
            G_in = electric_field_linear_term_n_direction(
                G_in, lattice, new_n, direction, lattice.dynamical_links_list
            )
            G.add_hamiltonians(G_in)

    # background string contribution
    for direction in [1, 2]:
        link_out = ((x, y), direction)
        if link_out in lattice.gauss_law_background:
            G.add_term('I' * lattice.n_qubits, -lattice.gauss_law_background[link_out])

        if direction == 1:
            link_in = ((x - 1, y), 1)
        else:
            link_in = ((x, y - 1), 2)

        if link_in in lattice.gauss_law_background:
            G.add_term('I' * lattice.n_qubits, +lattice.gauss_law_background[link_in])

    # uniform boundary field
    rho = boundary_flux_at_site(lattice, indices)
    G.add_term('I' * lattice.n_qubits, -rho)

    # dynamical staggered charge
    Q = charge_n_hamiltonian(lattice, indices)
    Q.hamiltonian = multiply_hamiltonian_by_constant(Q.hamiltonian, -1.0)
    G.add_hamiltonians(Q)

    G.cleanup()
    return G

def gauss_hamiltonian(lattice):
    H = Hamiltonian(lattice.n_qubits)

    for n in range(lattice.n_fermion_qubits):
        G = gauss_operator_n(lattice, n)

        indices = lattice.labels[n]
        if indices == lattice.charge_site:
            G.add_term('I'*lattice.n_qubits, -1.0)
        if indices == lattice.anticharge_site:
            G.add_term('I'*lattice.n_qubits, +1.0)

        G.multiply_hamiltonians(G)
        H.add_hamiltonians(G)

    return H

def gauss_hamiltonian_linear(lattice):
    H = Hamiltonian(lattice.n_qubits)

    for n in range(lattice.n_fermion_qubits):
        G = gauss_operator_n(lattice, n)

        indices = lattice.labels[n]
        #if indices == lattice.charge_site:
        #    G.add_term('I'*lattice.n_qubits, -1.0)
        #if indices == lattice.anticharge_site:
        #    G.add_term('I'*lattice.n_qubits, +1.0)
            
        H.add_hamiltonians(G)

    return H
    
def charge_total_hamiltonian_quadratic(lattice):
    total_charge = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            ch_n = charge_n_hamiltonian(lattice, (x, y))
            ch_n.multiply_hamiltonians(ch_n)
            total_charge.add_hamiltonians(ch_n)
    return total_charge

def site_z(lattice, n):
    I_term = list('I'*lattice.n_qubits)
    hamiltonian = Hamiltonian(lattice.n_qubits)
    term = copy.copy(I_term)
    term[lattice.n_gauge_qubits + n] = 'Z'
    hamiltonian.add_term(''.join(term), 1.0)
    return hamiltonian
    
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
    GL_hamiltonian_total = Hamiltonian(lattice.n_qubits)

    # =======================================================================================================================
    # Mass term
    for n in range(lattice.n_fermion_qubits):
        mass_hamiltonian = Hamiltonian(lattice.n_qubits)
        mass_hamiltonian = mass_term_n(mass_hamiltonian, lattice, n)
        mass_hamiltonian_total.add_hamiltonians(mass_hamiltonian)

    mass_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(mass_hamiltonian_total.hamiltonian, mass_coeff)
    mass_hamiltonian_total.cleanup()
    # =======================================================================================================================
    # print("Mass done")
    # =======================================================================================================================
    # Electric field term
    for n in range(lattice.n_fermion_qubits):
        electric_hamiltonian = Hamiltonian(lattice.n_qubits)
        electric_hamiltonian = electric_field_term_n(electric_hamiltonian, lattice, n, lattice.dynamical_links_list)
        electric_hamiltonian_total.add_hamiltonians(electric_hamiltonian)

    electric_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(electric_hamiltonian_total.hamiltonian, electric_coeff)
    electric_hamiltonian_total.cleanup()
    # =======================================================================================================================
    # print("Ele. done")
    # =======================================================================================================================
    # Magnetic field term
    for n in lattice.plaquettes:
        magnetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        magnetic_hamiltonian = magnetic_term_n(magnetic_hamiltonian, lattice, n, lattice.dynamical_links_list)
        magnetic_hamiltonian_total.add_hamiltonians(magnetic_hamiltonian)

    magnetic_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(magnetic_hamiltonian_total.hamiltonian, magnetic_coeff)
    magnetic_hamiltonian_total.cleanup()
    # =======================================================================================================================
    # print("Mag. done")
    # =======================================================================================================================
    # Kinetic term
    for n in range(lattice.n_fermion_qubits):
        kinetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        kinetic_hamiltonian = kinetic_term_n(kinetic_hamiltonian, lattice, n, lattice.dynamical_links_list)
        kinetic_hamiltonian_total.add_hamiltonians(kinetic_hamiltonian)

    kinetic_hamiltonian_total.hamiltonian = multiply_hamiltonian_by_constant(kinetic_hamiltonian_total.hamiltonian, kinetic_coeff)
    kinetic_hamiltonian_total.cleanup()
    # =======================================================================================================================
    # print("Kin. done")   

    full_hamiltonian.add_hamiltonians(mass_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(electric_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(magnetic_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(kinetic_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(BEF_hamiltonian_total)
    full_hamiltonian.add_hamiltonians(GL_hamiltonian_total)
    
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