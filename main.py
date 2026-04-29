from modules import *
from classes import *
from running_helpers import *
from QED_hamiltonian import *
from my_spsa import *

parameters = {
    'L_x': 4,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers': 2,
    'n_extra_layers': 0,
    'shots': 10000,
    'm': 3.0,
    'g': 0.25,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [((0,0), 1), ((1,0), 2), ((2,0), 2)],
    'gauss_weight': 0.0, # Enforces Gauss' law
    'charge_weight': 10000.0,
    'simulator': CPU_NOISELESS_SIMULATOR,
    'noisy' : False
}
    
SPSA_parameters = {
    'max_iters' : 10000,
    'average_length' : 5,
    'grad_tol' : 1e-12,
    'average_tol' : 1e-10,
    'a' : 0.08,
    'c' : 0.03,
    'prints' : False,
    'diagnostics' : False
    }

def statevector_solver(parameters, lattice):
    hamiltonian = generate_qed_hamiltonian(parameters, lattice)
    H = hamiltonian.to_sparse_matrix()
    H = H.astype(np.complex128)

    evals, evecs = eigsh(H, k=1, which='SA')
    
    E0 = evals[0].real
    psi_vec = evecs[:, 0]

    return E0, psi_vec

gs = [3.0]
lambdas = np.linspace(0.5, 10.0, 10)
E_0s = np.linspace(0.0, 20.0, 150)

E_0_mins = [0.0, 1.0, 2.2, 3.0, 4.4, 5.5, 6.2]
E_0_maxs = [4.0, 5.0, 6.2, 7.0, 8.4, 9.5, 10.2]

peak_total = []
counter = 0
E_stars = []
chi_max_heights_total = []
for g in gs:
    parameters['g'] = g
    print("# g =", parameters['g'])

    particle_numbers_all = []
    E_star = []
    chi_max_heights = []
    lamk = 0
    for lam in lambdas:
        parameters['m'] = lam * (parameters['g'] ** 2)
        particle_numbers = []
        # E_0s = np.linspace(E_0_mins[lamk], E_0_maxs[lamk], 100)
        gap = E_0s[1] - E_0s[0]
        for e_0 in E_0s:
            lattice = Lattice(
                parameters['L_x'],
                parameters['L_y'],
                parameters['gauge_truncation'],
                parameters['dynamical_links'],
                charge_site=(),
                anticharge_site=(),
                E_0=[e_0, e_0]
            )

            val, vec = statevector_solver(parameters, lattice)
            results = observes_reduced_from_statevector(vec, lattice, to_print=False)

            particle_number_total = results['particle_number_total']
            particle_numbers.append(particle_number_total)

        particle_numbers_all.append(particle_numbers)
        lamk += 1
        chi = []
        for k in range(1, len(particle_numbers) - 1):
            chi.append((particle_numbers[k + 1] - particle_numbers[k - 1]) / (2 * gap))
        chi_max_loc = np.argmax(np.abs(chi))
        chi_max_height = np.max(np.abs(chi))
        E_star.append(E_0s[chi_max_loc + 1])
        chi_max_heights.append(chi_max_height)
    
    print("E_star_"+str(counter)+" =", E_star)
    print("chi_max_height"+str(counter)+" =",chi_max_heights)
    chi_max_heights_total.append(chi_max_heights)
    E_stars.append(E_star)
    counter += 1
print("lambdas =",lambdas)
print("E_stars =",E_stars)
print("chi_max_heights=",chi_max_heights_total)