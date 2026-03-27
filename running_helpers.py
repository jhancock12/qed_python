from modules import *
from classes import *
from QED_hamiltonian import *
 
def _normalize_statevector(psi_vec):
    psi = np.asarray(psi_vec, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Input statevector has zero norm.")
    return psi / norm


def electric_field_values_from_statevector(psi_vec, lattice):
    psi = _normalize_statevector(psi_vec)
    values = {}

    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            H = Hamiltonian(lattice.n_qubits)
            H = electric_field_linear_term_n_direction(
                H, lattice, n, direction, lattice.dynamical_links_list
            )
            H_sparse = H.to_sparse_matrix()
            values[(lattice.labels[n], direction)] = np.real(np.vdot(psi, H_sparse @ psi))

    return values

def electric_field_values_squared_from_statevector(psi_vec, lattice):
    psi = _normalize_statevector(psi_vec)
    values = {}

    for n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[n]:
            H = Hamiltonian(lattice.n_qubits)
            H = electric_field_linear_term_n_direction(
                H, lattice, n, direction, lattice.dynamical_links_list
            )
            H.multiply_hamiltonians(H)
            H_sparse = H.to_sparse_matrix()
            values[(lattice.labels[n], direction)] = np.real(np.vdot(psi, H_sparse @ psi))

    return values

def magnetic_field_values_from_statevector(psi_vec, lattice):
    psi = _normalize_statevector(psi_vec)
    values = {}

    for n in lattice.plaquettes:
        H = Hamiltonian(lattice.n_qubits)
        H = magnetic_term_n(H, lattice, n, lattice.dynamical_links_list)
        H_sparse = H.to_sparse_matrix()
        values[lattice.labels[n]] = np.real(np.vdot(psi, H_sparse @ psi))

    return values
    
def charge_values_from_statevector(psi_vec, lattice):
    psi = _normalize_statevector(psi_vec)
    values = {}

    for n in range(lattice.n_fermion_qubits):
        indices = lattice.labels[n]
        H = charge_n_hamiltonian(lattice, indices)
        H_sparse = H.to_sparse_matrix()
        values[indices] = np.real(np.vdot(psi, H_sparse @ psi))

    return values

def particle_number_values_from_statevector(psi_vec, lattice):
    psi = _normalize_statevector(psi_vec)
    values = {}

    for x in range(lattice.L_x):
        for y in range(lattice.L_y):
            H = particle_n_hamiltonian(lattice, (x, y))
            H_sparse = H.to_sparse_matrix()
            values[(x, y)] = np.real(np.vdot(psi, H_sparse @ psi))

    return values
    
def observe_and_print_reduced_from_statevector(
    psi_vec, lattice
):
    gauss_equations = lattice.gauss_equations
    ef = electric_field_values_from_statevector(psi_vec, lattice)
    ef_sq = electric_field_values_squared_from_statevector(psi_vec, lattice)
    mf = magnetic_field_values_from_statevector(psi_vec, lattice)
    pn = particle_number_values_from_statevector(psi_vec, lattice)
    c = charge_values_from_statevector(psi_vec, lattice)

    total_pn = sum(pn.values())

    for dependant_variable in gauss_equations['dependant_variables']:
        equation = gauss_equations['solution'][dependant_variable]
        sub_ins = {}
        for var in gauss_equations['independant_variables']:
            if var in list(gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[gauss_equations['reverse_link_variable_dict'][var]]
            else:
                n_charge = int(str(var)[1:])
                sub_ins[var] = c[lattice.labels[n_charge]]
        ef[gauss_equations['reverse_link_variable_dict'][dependant_variable]] = complex(equation.subs(sub_ins))

    n = 0
    gl = {}
    for equation in gauss_equations['solution']:
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
    ef_sq = smart_round(ef_sq, 6)
    mf = smart_round(mf, 6)
    pn = smart_round(pn, 6)
    c = smart_round(c, 6)

    total_pn = smart_round(total_pn, 6)

    print("electric_field_values_dict:", ef)
    print("charge_field_values_dict:", c)
    print("magnetic_field_values_dict:", mf)
    print("particle_number_value_dict:", pn)
    print("total_particle_number:", total_pn)

    return {
        'electric_field_dict': ef,
        'electric_field_squared_dict': ef_sq,
        'charge_field_dict': c,
        'magnetic_field_dict': mf,
        'particle_number_dict': pn,
        'particle_number_total': total_pn
    }