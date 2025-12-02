from classes import *

def Es_function(qubits_per_gauge):
    I_string_list = list('I' * qubits_per_gauge)
    coeff = -0.5
    Es = {}
    
    first_string = copy.copy(I_string_list)
    first_string[0] = 'Z'
    Es["".join(first_string)] = coeff * (2**(qubits_per_gauge - 1) - 1)

    for k in range(qubits_per_gauge - 1):
        temp_string = copy.copy(I_string_list)
        temp_string[qubits_per_gauge - k - 1] = 'Z'
        Es["".join(first_string)] = 2**(k)
    return Es
    

def mass_term_n(hamiltonian, lattice, n): 
    gauge_string = 'I'*lattice.n_dynamical_gauge_qubits
    fermion_before = 'I'*n
    fermion_after = 'I'*(lattice.n_fermion_qubits - n - 1)

    coordinates = lattice.get_coordinates(n)
    term = gauge_string + fermion_before + 'Z' + fermion_after
    coeff = ((-1)**(coordinates[0] + coordinates[1])) / 2

    hamiltonian.add_term(term, coeff)
    hamiltonian.add_term('I'*lattice.n_qubits, coeff)

    return hamiltonian

def electric_field_term_n_direction(hamiltonian, lattice, n, direction, dynamical_links):
    index = lattice.labels[n] 
    if (index, direction) in dynamical_links:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * ((link_index) * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)

        Es = Es_function(lattice.qubits_per_gauge)

        for key in list(Es.keys()):
            term = gauge_before + key + gauge_after + fermion_string
            coeff = Es[key]
            hamiltonian.add_term(term, coeff)
        hamiltonian.multiply_hamiltonians(hamiltonian)
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 0)

    return hamiltonian

def electric_field_linear_term_n_direction(hamiltonian, lattice, n, direction, dynamical_links):
    index = lattice.labels[n]
    if (index, direction) in dynamical_links:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * ((link_index) * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)

        Es = Es_function(lattice.qubits_per_gauge)

        for key in list(Es.keys()):
            term = gauge_before + key + gauge_after + fermion_string
            coeff = Es[key]
            hamiltonian.add_term(term, coeff)
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 0)

    return hamiltonian

def electric_field_term_n(hamiltonian, lattice, n, dynamical_links):
    E_temp = Hamiltonian(lattice.n_qubits)
    for direction in lattice.directions[n]:
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

        Us = {
                2: {
            "XI": 1.0,
            "XX": 1.0,
            "YI": -1.0j,
            "YX": 1.0j
            },        
        
                3:  { # This looks so wrong - not using 3 per gauge atm
            "IIX": 2.0,
            "IXX": 1.0,
            "XXX": 1.0,
            "IYX": 1.0j, 
            "XYX": 1.0j,
            "YII": 2.0j,
            "YXI": 1.0j,
            "YXX": 1.0j,
            "YYI": 1.0,
            "YYX": -1.0
            },
            
                4 : {
            "IIIX": 4.0,      # 4 X0
            "IIXX": 2.0,      # 2 X0 X1
            "IXXX": 1.0,      # X0 X1 X2
            "XXXX": 1.0,      # X0 X1 X2 X3
        
            "IXXY": 1.0j,     # i X0 X1 Y2
            "XXXY": 1.0j,    # i X0 X1 Y2 X3  (string below is correct)
        
            "IIYX": 2.0j,     # i*2 X0 Y1
            "IXYX": 1.0j,     # i X0 Y1 X2
            "XXYX": 1.0j,     # i X0 Y1 X2 X3
        
            "IYYX": 1.0,      # X0 Y1 Y2
            "XYYX": -1.0,     # - X0 Y1 Y2 X3
        
            "IIIY": 4.0j,     # i*4 Y0
            "IIXY": 2.0j,     # i*2 Y0 X1
            "IXXY": 1.0j,     # i Y0 X1 X2
            "XXXY": 1.0j,    # i Y0 X1 X2 X3 
        
            "IXYY": 1.0,      # Y0 X1 Y2
            "XXYY": -1.0,     # - Y0 X1 Y2 X3
        
            "IYYY": 2.0,      # 2 Y0 Y1
            "IXYY": -1.0,     # - Y0 Y1 X2
            "XXYY": -1.0,     # - Y0 Y1 X2 X3
        
            "IYYY": 1.0j,     # i Y0 Y1 Y2
            "XYYY": 1.0j      # i Y0 Y1 Y2 X3
            }
            }
        for key in list(Us[lattice.qubits_per_gauge]):
            term = gauge_before + key + gauge_after + fermion_string
            value = Us[lattice.qubits_per_gauge][key]
            hamiltonian.add_term(term, value*(1/(2**(lattice.qubits_per_gauge-1))))  
        
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
    Us[0].hamiltonian = Us[0].to_conjugate()
    Us[1].hamiltonian = Us[1].to_conjugate()

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
    gauge_string = 'I'*lattice.n_dynamical_gauge_qubits
    fermions_before = 'Z'*(n)
    fermions_after = 'I'*(lattice.n_fermion_qubits - (n+1))
    hamiltonian.add_term(gauge_string + fermions_before + 'X' + fermions_after, (1j)**(n-1)/2)
    hamiltonian.add_term(gauge_string + fermions_before + 'Y' + fermions_after, (-1j)*(1j)**(n-1)/2)
    return hamiltonian

def annihilation_operator_n(hamiltonian, lattice, n):
    gauge_string = 'I'*lattice.n_dynamical_gauge_qubits
    fermions_before = 'Z'*(n)
    fermions_after = 'I'*(lattice.n_fermion_qubits - (n+1))
    hamiltonian.add_term(gauge_string + fermions_before + 'X' + fermions_after, (-1j)**(n-1)/2)
    hamiltonian.add_term(gauge_string + fermions_before + 'Y' + fermions_after, (1j)*(-1j)**(n-1)/2)
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

    hamiltonian.multiply_hamiltonians(annihilation_hamiltonian)
    hamiltonian.multiply_hamiltonians(U_hamiltonian)
    hamiltonian.multiply_hamiltonians(creation_hamiltonian)

    return hamiltonian

def kinetic_term_n(hamiltonian, lattice, n, dynamical_links):
    # This works out from both directions for a given n
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
 
def generate_qed_hamiltonian(parameters):
    lattice = Lattice(parameters['L_x'],parameters['L_y'],parameters['gauge_truncation'],parameters['dynamical_links'])
    full_hamiltonian = Hamiltonian(lattice.n_qubits)
    
    mass_coeff = parameters['m']
    electric_coeff = parameters['g']*parameters['g']/2
    magnetic_coeff = -1/(2*(parameters['a']*parameters['a'])*(parameters['g']*parameters['g']))
    kinetic_coeff = 1/(2*parameters['a'])

    # For testing
    mass_coeff *= 1
    electric_coeff *= 1
    magnetic_coeff *= 1
    kinetic_coeff *= 1

    # Mass term
    for n in range(lattice.n_fermion_qubits):
        mass_hamiltonian = Hamiltonian(lattice.n_qubits)
        mass_hamiltonian = mass_term_n(mass_hamiltonian, lattice, n)
        mass_hamiltonian.hamiltonian = multiply_hamiltonian_by_constant(mass_hamiltonian.hamiltonian, mass_coeff)

        full_hamiltonian.add_hamiltonians(mass_hamiltonian)
    
    full_hamiltonian.cleanup()
    
    # Electric field term
    for n in range(lattice.n_fermion_qubits):
        electric_hamiltonian = Hamiltonian(lattice.n_qubits)
        electric_hamiltonian = electric_field_term_n(electric_hamiltonian, lattice, n, parameters['dynamical_links'])
        electric_hamiltonian.hamiltonian = multiply_hamiltonian_by_constant(electric_hamiltonian.hamiltonian, electric_coeff)

        full_hamiltonian.add_hamiltonians(electric_hamiltonian)
    
    full_hamiltonian.cleanup()
    
    # Magnetic field term
    for n in lattice.plaquettes:
        magnetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        magnetic_hamiltonian = magnetic_term_n(magnetic_hamiltonian, lattice, n, parameters['dynamical_links'])
        magnetic_hamiltonian.hamiltonian = multiply_hamiltonian_by_constant(magnetic_hamiltonian.hamiltonian, magnetic_coeff)

        full_hamiltonian.add_hamiltonians(magnetic_hamiltonian)
    
    full_hamiltonian.cleanup()
    
    # Kinetic term
    for n in range(lattice.n_fermion_qubits):
        kinetic_hamiltonian = Hamiltonian(lattice.n_qubits)
        kinetic_hamiltonian = kinetic_term_n(kinetic_hamiltonian, lattice, n, parameters['dynamical_links'])
        kinetic_hamiltonian.hamiltonian = multiply_hamiltonian_by_constant(kinetic_hamiltonian.hamiltonian, kinetic_coeff)

        full_hamiltonian.add_hamiltonians(kinetic_hamiltonian)

    full_hamiltonian.cleanup()

    # Background electric field term
    for n in range(lattice.n_fermion_qubits):
        background_electric_hamiltonian = Hamiltonian(lattice.n_qubits)
        background_electric_hamiltonian = background_electric_field_term_n(electric_hamiltonian, lattice, n, parameters['dynamical_links'])
        background_electric_hamiltonian.hamiltonian = multiply_hamiltonian_by_constant(electric_hamiltonian.hamiltonian, 2*parameters['E_0']) 
        full_hamiltonian.add_hamiltonians(background_electric_hamiltonian) # This adds the 2E_0\hat{E} term for each n in both directions
    background_electric_hamiltonian = Hamiltonian(lattice.n_qubits)    
    background_electric_hamiltonian.add_term('I'*lattice.n_qubits, parameters['E_0']*parameters['E_0']*len(parameters['dynamical_links'])) # Adds the E_0^2 term for each dynamical link
    background_electric_hamiltonian.add_term('I'*lattice.n_qubits, (2*parameters['E_0'] + parameters['E_0']*parameters['E_0']) * (lattice.n_links-len(parameters['dynamical_links']))) # For each of the non-dynamical links
    full_hamiltonian.add_hamiltonians(background_electric_hamiltonian)
    
    # Do I need to add backgrounds for non-dynamical links? Each of these is set to I, and so I should add (N_{TL}-N_{DL})(2E_0 + E_0^2)?
    
    full_hamiltonian.cleanup()

    return full_hamiltonian

def particle_n_hamiltonian(lattice, coordinates):
    n = lattice.get_index(coordinates[0], coordinates[1])
    pn_hamiltonian = Hamiltonian(lattice.n_qubits)
    
    gauge_string = 'I'*lattice.n_dynamical_gauge_qubits
    fermion_before = 'I'*n
    fermion_after = 'I'*(lattice.n_fermion_qubits - n - 1)
    Z_term = gauge_string + fermion_before + 'Z' + fermion_after
    coefficient = 1 if (coordinates[0] + coordinates[1]) % 2 == 0 else -1
    pn_hamiltonian.add_term(Z_term, coefficient / 2.0)
    
    return pn_hamiltonian

def particle_number_hamiltonian(lattice):
    P_n = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            p_n = Hamiltonian(lattice.n_qubits)
            p_n = particle_n_hamiltonian(lattice, (x, y))
            P_n.add_hamiltonians(p_n)
    return P_n

def charge_n_hamiltonian(lattice, coordinates):
    j = lattice.get_index(coordinates[0], coordinates[1])
    charge_hamiltonian = Hamiltonian(lattice.n_qubits)
    I_term = 'I' * lattice.n_qubits
    Z_term = I_term[:lattice.n_gauge_qubits+j] + 'Z' + I_term[lattice.n_gauge_qubits+j+1:]
    
    coefficient = 1 if (coordinates[0] + coordinates[1]) % 2 == 0 else -1
    charge_hamiltonian.add_term(I_term, coefficient)
    charge_hamiltonian.add_term(Z_term, -1)
    
    return charge_hamiltonian
    
def charge_total_hamiltonian(lattice):
    total_charge = Hamiltonian(lattice.n_qubits)
    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            charg_n = Hamiltonian(lattice.n_qubits)
            charg_n = charge_n_hamiltonian(lattice, (x, y))
            total_charge.add_hamiltonians(charg_n)
    return total_charge