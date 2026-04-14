from modules import *
from classes import *
from running_helpers import *
from QED_hamiltonian import *
from my_spsa import *

parameters = {
    'L_x': 3,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers': 2,
    'n_extra_layers': 0,
    'shots': 10000,
    'm': 3.0,
    'g': 1.0,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [((0,0), 1), ((1,0), 2)],
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

gs = [1.0]
ms = np.linspace(1.0, 5.0, 5)
E_0s = np.linspace(0.0, 15.0, 60)
gap = E_0s[1] - E_0s[0]
peak_total = []

dynamical_links_sets = [
    [((0,0), 1), ((1,0), 1)],
    [((0,0), 1), ((1,0), 2)],
    [((0,0), 1), ((2,0), 2)],
    [((0,0), 1), ((1,1), 1)],
    [((0,0), 2), ((1,0), 1)],
    [((0,0), 2), ((1,0), 2)],
    [((0,0), 2), ((2,0), 2)],
    [((0,0), 2), ((1,1), 1)],
    [((1,0), 1), ((1,0), 2)],
    [((1,0), 1), ((0,1), 1)],
    [((1,0), 2), ((2,0), 2)],
    [((1,0), 2), ((0,1), 1)],
    [((1,0), 2), ((1,1), 1)],
    [((2,0), 2), ((0,1), 1)],
    [((0,1), 1), ((1,1), 1)]
]

for dynamical_links in dynamical_links_sets:
    print("\n")
    print("#--- dynamical links:",dynamical_links,"---")
    print("#--- both-directions ---")
    peak_total = []
    parameters['dynamical_links'] = dynamical_links
    for m in ms:
        parameters['m'] = m
        peak_row = []
        for g in gs:
            parameters['g'] = g
            particle_numbers = []
            for e_0 in E_0s:
                lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [e_0, e_0])
                val, vec = statevector_solver(parameters, lattice)
                results = observes_reduced_from_statevector(vec, lattice, to_print = False)
            
                particle_number_total = results['particle_number_total']
    
                particle_numbers.append(particle_number_total)
                # print("g =",g,", m =",m,"E_0 =",e_0,"done!")
    
            particle_chi = []
            E_0s_chi = []
    
            for k in range(1, len(particle_numbers) - 1):
                chi = (particle_numbers[k + 1] - particle_numbers[k - 1]) / (2 * gap)
                particle_chi.append(chi)
                E_0s_chi.append(E_0s[k])
            peak_point = E_0s_chi[np.argmax(particle_chi)]
    
            peak_row.append(peak_point)
    
            
        peak_total.append(peak_row)
    
    peak_total = list(np.array(peak_total).flatten())
    
    print("peak_total_both_small=",peak_total)
    
    print("#--- x-directions ---")
    peak_total = []
    for m in ms:
        parameters['m'] = m
        peak_row = []
        for g in gs:
            parameters['g'] = g
            particle_numbers = []
            for e_0 in E_0s:
                lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [e_0, 0.0])
                val, vec = statevector_solver(parameters, lattice)
                results = observes_reduced_from_statevector(vec, lattice, to_print = False)
            
                particle_number_total = results['particle_number_total']
    
                particle_numbers.append(particle_number_total)
                # print("g =",g,", m =",m,"E_0 =",e_0,"done!")
    
            particle_chi = []
            E_0s_chi = []
    
            for k in range(1, len(particle_numbers) - 1):
                chi = (particle_numbers[k + 1] - particle_numbers[k - 1]) / (2 * gap)
                particle_chi.append(chi)
                E_0s_chi.append(E_0s[k])
            peak_point = E_0s_chi[np.argmax(particle_chi)]
    
            peak_row.append(peak_point)
    
            
        peak_total.append(peak_row)
    
    peak_total = list(np.array(peak_total).flatten())
    
    print("peak_total_x_small=",peak_total)
    
    print("#--- y-directions ---")
    peak_total = []
    for m in ms:
        parameters['m'] = m
        peak_row = []
        for g in gs:
            parameters['g'] = g
            particle_numbers = []
            for e_0 in E_0s:
                lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [0.0, e_0])
                val, vec = statevector_solver(parameters, lattice)
                results = observes_reduced_from_statevector(vec, lattice, to_print = False)
            
                particle_number_total = results['particle_number_total']
    
                particle_numbers.append(particle_number_total)
                # print("g =",g,", m =",m,"E_0 =",e_0,"done!")
    
            particle_chi = []
            E_0s_chi = []
    
            for k in range(1, len(particle_numbers) - 1):
                chi = (particle_numbers[k + 1] - particle_numbers[k - 1]) / (2 * gap)
                particle_chi.append(chi)
                E_0s_chi.append(E_0s[k])
            peak_point = E_0s_chi[np.argmax(particle_chi)]
    
            peak_row.append(peak_point)
    
            
        peak_total.append(peak_row)
    
    peak_total = list(np.array(peak_total).flatten())
    
    print("peak_total_y_small=",peak_total)