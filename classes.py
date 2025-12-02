from modules import *

def multiply_pauli_strings(p1, p2):
    result = []
    phase = 1+0j
    for a, b in zip(p1, p2):
        res, ph = PAULI_PHASES[a][b]
        result.append(res)
        phase *= ph
    return "".join(result), phase   

def multiply_hamiltonian_by_constant(hamiltonian, c):
    new_hamil = {}
    for term, coeff in hamiltonian.items():
        new_hamil[term] = c * coeff
    return new_hamil

def possible_directions(lattice):
    labels = lattice.labels
    possible_labels = list(lattice.reverse_labels.keys())
    directions = {}
    link_indexing = {}
    dynamical_link_indexing = {}
    counter = 0
    dynamical_counter = 0
    for j in range(len(labels)):
        directions[j] = []
        if (labels[j][0] + 1, labels[j][1]) in possible_labels: # x_link
            directions[j].append(1)
            link_indexing[((labels[j][0], labels[j][1]), 1)] = counter
            counter += 1
        if (labels[j][0], labels[j][1]+1) in possible_labels: # y_link
            directions[j].append(2)
            link_indexing[((labels[j][0], labels[j][1]), 2)] = counter
            counter += 1
        
        if ((labels[j][0], labels[j][1]), 1) in lattice.dynamical_links_list:
            dynamical_link_indexing[((labels[j][0], labels[j][1]), 1)] = dynamical_counter
            dynamical_counter += 1
        if ((labels[j][0], labels[j][1]), 2) in lattice.dynamical_links_list:
            dynamical_link_indexing[((labels[j][0], labels[j][1]), 2)] = dynamical_counter
            dynamical_counter += 1

    return directions, link_indexing, dynamical_link_indexing

def possible_plaquettes(lattice):
    plaquette_ns = []
    indices_list = list(lattice.reverse_labels.keys())
    for key in lattice.labels:
        indices = lattice.labels[key]
        if (indices[0], indices[1] + 1) in indices_list:
            if (indices[0] + 1, indices[1]) in indices_list:
                if (indices[0] + 1, indices[1] + 1) in indices_list:
                    plaquette_ns.append(key)
    return plaquette_ns

class Lattice:
    def __init__(self, L_x, L_y, gauge_truncation, dynamical_links_list):

        # Need to add link indexing

        if L_x*L_y % 2 == 1:
            raise ValueError(f"L_x * L_y must be even")    
        self.L_x = L_x
        self.L_y = L_y
        self.labels = {}
        self.reverse_labels = {}
        self.gauge_truncation = gauge_truncation
        self.dynamical_links_list = dynamical_links_list
        counter = 0

        for y in range(L_y):
            for x in range(L_x):
                self.labels[counter] = (x, y)
                self.reverse_labels[(x, y)] = counter
                counter += 1
        self.n_fermion_qubits = counter
        self.n_links = (self.L_x - 1) * self.L_y + self.L_x * (self.L_y - 1) # OBC 
        self.qubits_per_gauge = 0 if gauge_truncation == 0 else int(np.ceil(np.log2(2*gauge_truncation+1)))
        self.n_gauge_qubits = self.n_links*self.qubits_per_gauge
    
        self.optimal_n_dynamical_links = self.n_links - (self.n_fermion_qubits - 1)
        self.n_dynamical_links = len(self.dynamical_links_list)
        
        if self.n_dynamical_links < self.optimal_n_dynamical_links:
            raise ValueError(f"This lattice requires at least {self.optimal_n_dynamical_links} dynamical links, you gave: {self.n_dynamical_links}")

        self.n_dynamical_gauge_qubits = self.n_dynamical_links*self.qubits_per_gauge
        self.n_qubits = self.n_fermion_qubits + self.n_dynamical_gauge_qubits

        self.directions, self.link_indexing, self.dynamical_link_indexing = possible_directions(self)
        self.plaquettes = possible_plaquettes(self)

    def get_index(self, x, y):
        if (x, y) in self.reverse_labels:
            return self.reverse_labels[(x, y)]
        raise ValueError(f"Site ({x}, {y}) is out of bounds.")

    def get_coordinates(self, index):
        if index in self.labels:
            return self.labels[index]
        raise ValueError(f"Index {index} is out of bounds.")

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L_x and 0 <= ny < self.L_y:
                neighbors.append((nx, ny))
        return neighbors
    
class Hamiltonian:
    def __init__(self, n_qubits = 1):
        self.hamiltonian = {'I'*n_qubits : 0}
        self.n_qubits = n_qubits

    def latex_print(self):
        keys = list(self.hamiltonian.keys())
        string_to_print = ""
        for key in keys:
            matrix_list = list(key)
            if matrix_list == ['I']*len(matrix_list):
                string_to_print += str(self.hamiltonian[key])
            else:
                temp_string = ""
                for i in range(len(matrix_list)):
                    if matrix_list[i] != 'I':
                        temp_string += matrix_list[i] + r"_{" + str(i) + r"} "    
                              
                if self.hamiltonian[key] < 0:
                    string_to_print += r" - " + str(abs(self.hamiltonian[key])) + " " + temp_string
                else:
                    string_to_print += r" + " + str(abs(self.hamiltonian[key])) + " " + temp_string
        print(string_to_print)

    def latex_plot(self, save=False):
        keys = list(self.hamiltonian.keys())
        string_to_print = "H = &"
        counter = 0
        for key in keys:
            matrix_list = list(key)
            if matrix_list == ['I']*len(matrix_list):
                term = str(self.hamiltonian[key])
                if self.hamiltonian[key] < 0:
                    string_to_print += r" - " + term
                else:
                    string_to_print += r" + " + term
            else:
                temp_string = ""
                for i in range(len(matrix_list)):
                    if matrix_list[i] != 'I':
                        temp_string += matrix_list[i] + r"_{" + str(i) + r"} "    
                            
                if self.hamiltonian[key] < 0:
                    string_to_print += r" - " + str(abs(self.hamiltonian[key])) + " " + temp_string
                else:
                    string_to_print += r" + " + str(abs(self.hamiltonian[key])) + " " + temp_string
                counter += 1
                if (counter % 4) == 0:
                    string_to_print += r"\\ &" 
        
        # Remove leading " + " if present
        if string_to_print.startswith(" + "):
            string_to_print = string_to_print[3:]
        elif string_to_print.startswith(" - "):
            string_to_print = string_to_print[1:]  # Keep the minus but remove space
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"  # ADD THIS LINE
        })

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # Use text with wrapping
        ax.text(0.5, 0.5, r"$\begin{aligned}" + string_to_print + r"\end{aligned}$", 
                fontsize=14, ha='center', va='center')
                 
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if save:
            plt.savefig(f"equation_temp.pdf", bbox_inches='tight')
        plt.show()
             
    def cleanup(self):
        new_hamil = {}
        for key in self.hamiltonian:
            if np.abs(self.hamiltonian[key]) > 0.0001:
                new_hamil[key] = self.hamiltonian[key]
                if self.hamiltonian[key].imag > 0.0001:
                    print("This Hamiltonian will be non-Hermitian, due to the term:", key, ":", self.hamiltonian[key])
                else:
                    new_hamil[key] = new_hamil[key].real
        self.hamiltonian = new_hamil
    
    def multiply_hamiltonians(self, other):
        hamil_1 = self.hamiltonian
        if isinstance(other, Hamiltonian): hamil_2 = other.hamiltonian
        else: hamil_2 = other
        new_hamil = {}
        for term1, coeff1 in hamil_1.items():
            for term2, coeff2 in hamil_2.items():
                new_term, phase = multiply_pauli_strings(term1, term2)
                new_coeff = coeff1 * coeff2 * phase
                if abs(new_coeff.imag) < 1e-12:
                    new_coeff = new_coeff.real
                if new_term in new_hamil:
                    new_hamil[new_term] += new_coeff
                else:
                    new_hamil[new_term] = new_coeff
        self.hamiltonian = new_hamil

    def add_term(self, term, coeff):
        if len(term) != self.n_qubits:
            raise ValueError(f"Terms must be of length {self.n_qubits} for this Hamiltonian, you have entered a term of length {len(term)}")
        else:
            if term in self.hamiltonian.keys():
                self.hamiltonian[term] += coeff
            else:
                self.hamiltonian[term] = coeff
    
    def add_hamiltonians(self, other):
        if isinstance(other, Hamiltonian): hamil_other = other.hamiltonian
        else : hamil_other = other
        for key in hamil_other:
            self.add_term(key, hamil_other[key])
    
    def to_matrix(self):
        matrix_dict = {
            'I': np.array([[1,0],[0,1]]),
            'X': np.array([[0,1],[1,0]]),
            'Y': np.array([[0,-1j],[1j,0]]),
            'Z': np.array([[1,0],[0,-1]])
            }
        matrix = np.zeros((2**self.n_qubits,2**self.n_qubits), dtype=complex)
        terms = list(self.hamiltonian.keys())
        for term in terms:
            temp_matrix = np.array([1])
            for t in list(term):
                temp_matrix = np.kron(matrix_dict[t], temp_matrix)
            matrix += self.hamiltonian[term]*temp_matrix
        return matrix
    
    def to_sparse_matrix(self):
        
        I = spar.csc_matrix([[1, 0], [0, 1]], dtype=complex)
        X = spar.csc_matrix([[0, 1], [1, 0]], dtype=complex)
        Y = spar.csc_matrix([[0, -1j], [1j, 0]], dtype=complex)
        Z = spar.csc_matrix([[1, 0], [0, -1]], dtype=complex)

        matrix_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

        # start with a zero sparse matrix
        dim = 2 ** self.n_qubits
        H = spar.csc_matrix((dim, dim), dtype=complex)

        for term, coeff in self.hamiltonian.items():
            # Build kron product for this term
            temp = spar.csc_matrix([1], dtype=complex)

            for t in term:       # e.g. "XIZY"
                temp = spar.kron(matrix_dict[t], temp, format='csc')

            H += coeff * temp

        return H
    
    def to_conjugate(self):
        # Of note here, Pauli strings are self-conjugate
        hamil = self.hamiltonian
        keys = list(hamil.keys())
        new_hamil = {}
        for key in keys:
            new_hamil[key] = hamil[key].real - 1j*hamil[key].imag
        self.conjugate = new_hamil
        return new_hamil

class CircuitBuilder:
    def __init__(self, n_fermion_qubits, n_gauge_qubits):
        self.n_fermion_qubits = n_fermion_qubits
        self.n_gauge_qubits = n_gauge_qubits
        self.n_qubits = n_fermion_qubits + n_gauge_qubits
        self.circuit = qiskit.QuantumCircuit(self.n_qubits,self.n_qubits)
    
    def iSwap(self, theta, j, k):
        self.circuit.ryy(theta/2, j, k)
        self.circuit.rxx(theta/2, j, k)
        return self
    
    def iSwap_block_calculate(self):
        total_layers = int(self.n_fermion_qubits / 2)
        counter = 0
        for layer in range(total_layers):
            qubits_to_use = list(range(layer,self.n_fermion_qubits-layer))
            for _ in range(int(len(qubits_to_use)/2)):
                counter += 1
        return counter

    def iSwap_block(self,thetas_slice):
        total_layers = int(self.n_fermion_qubits / 2)
        counter = 0
        for layer in range(total_layers):
            qubits_to_use = list(range(layer,self.n_fermion_qubits-layer))
            for j in range(int(len(qubits_to_use)/2)):
                self.iSwap(thetas_slice[counter], self.n_gauge_qubits+qubits_to_use[0]+2*j, self.n_gauge_qubits+qubits_to_use[0]+2*j+1)
                counter += 1
        self.circuit.barrier()
        return self

    ## I know that this gauge section is not completely right yet
    # We would have to consider the direction of the connections that are being examined, i.e., U_{\mu,x}

    def gauge_gate_2(self, thetas, start_qubit):
        self.circuit.ry(thetas[0], start_qubit)
        self.circuit.cry(thetas[1], start_qubit, start_qubit+1)
        return self
    
    def gauge_gate_3(self,thetas,start_qubit):
        self.circuit.ry(thetas[0], start_qubit)
        self.circuit.ry(thetas[1], start_qubit+1)
        self.circuit.cry(thetas[2], start_qubit+1, start_qubit+2)
        self.circuit.cry(thetas[3], start_qubit, start_qubit+2)
        return self

    def gauge_block(self,thetas_slice,truncation):
        if truncation == 0:
            return self
        
        gauge_gates = {
            2: self.gauge_gate_2,
            3: self.gauge_gate_3
            }
        
        qubits_per_gauge = int(np.ceil(np.log2(2*truncation+1)))
        for j in range(int(len(thetas_slice)/qubits_per_gauge)):
            thetas = thetas_slice[qubits_per_gauge*j:qubits_per_gauge*j+(qubits_per_gauge)]
            gauge_gates[qubits_per_gauge](thetas,qubits_per_gauge*j)
        self.circuit.barrier()
        return self

    def initialize_fermions(self):
        for k in range(self.n_gauge_qubits // 2, self.n_qubits // 2):
            self.circuit.x(2*k + 1)
        self.circuit.barrier()
        return self
    
    def build(self):
        return self.circuit.copy()

class Measurements:
    def __init__(self, simulator=NOISELESS_SIMULATOR):
        self.simulator = simulator
    
    def measure_circuit(self, circuit, terms, shots):
        if terms == 'I' * len(terms):
            return {'0' * len(terms): shots}
        
        circuit_copy = circuit.copy()
        for j, term in enumerate(terms):
            if term == 'X':
                circuit_copy.h(j)
            elif term == 'Y':
                circuit_copy.sdg(j)
                circuit_copy.h(j)
            if term in ['X', 'Y', 'Z']:
                circuit_copy.measure(j, j)
        transpiled_circuit = qiskit.transpile(circuit_copy, self.simulator)
        job = self.simulator.run(transpiled_circuit, shots=shots)
        return job.result().get_counts(transpiled_circuit)
    
    def expected_value_from_counts(self, counts):
        total_shots = sum(counts.values())
        expected_value = 0
        for bitstring, count in counts.items():
            parity = 1 if bitstring.count('1') % 2 == 0 else -1
            expected_value += parity * count / total_shots
        return expected_value
    
    def expected_value_hamiltonian(self, hamiltonian, circuit, shots = 1024):
        expected_value_total = 0
        for terms, coefficient in hamiltonian.hamiltonian.items():
            counts = self.measure_circuit(circuit, terms, shots)
            expected_value_term = self.expected_value_from_counts(counts)
            expected_value_total += coefficient * expected_value_term
        return expected_value_total
    
class ObservableCalculator:
    def __init__(self, lattice, measurement_manager):
        #
        # Fermion qubits sit on the left end of the circuit
        #
        self.lattice = lattice
        self.measurer = measurement_manager
    
    def charge_n(self, coordinates, circuit, shots = 1024):
        j = self.lattice.get_index(coordinates[0], coordinates[1])
        charge_hamiltonian = Hamiltonian(self.lattice.n_qubits)
        I_term = 'I' * self.lattice.n_qubits
        Z_term = I_term[:self.lattice.n_gauge_qubits+j] + 'Z' + I_term[self.lattice.n_gauge_qubits+j+1:]
        
        coefficient = 1 if (coordinates[0] + coordinates[1]) % 2 == 0 else -1
        charge_hamiltonian.add_term(I_term, coefficient)
        charge_hamiltonian.add_term(Z_term, -1)
      
        return self.measurer.expected_value_hamiltonian(charge_hamiltonian, circuit, shots)
    
    def charge_total(self, circuit, shots = 1024):
        total_charge = 0
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                total_charge += self.charge_n((x, y), circuit, shots)
        return total_charge
    
    def particle_n(self, coordinates, circuit, shots = 1024):
        n = self.lattice.get_index(coordinates[0], coordinates[1])
        pn_hamiltonian = Hamiltonian(self.lattice.n_qubits)
        I_term = 'I' * self.lattice.n_qubits

        gauge_string = 'I'*self.lattice.n_dynamical_gauge_qubits
        fermion_before = 'I'*n
        fermion_after = 'I'*(self.lattice.n_fermion_qubits - n - 1)
        Z_term = gauge_string + fermion_before + 'Z' + fermion_after
        coefficient = 1 if (coordinates[0] + coordinates[1]) % 2 == 0 else -1
        pn_hamiltonian.add_term(Z_term, coefficient / 2.0)
      
        return self.measurer.expected_value_hamiltonian(pn_hamiltonian, circuit, shots)
    
    def particle_number(self, circuit, shots = 1024):
        p_n = self.lattice.n_fermion_qubits / 2.0
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                p_n += self.particle_n((x, y), circuit, shots)
        return p_n

    def energy(self, circuit, hamiltonian, shots = 1024):
        return self.measurer.expected_value_hamiltonian(hamiltonian, circuit, shots)
    
    def full_z(self, circuit, n, shots = 1024):
        return self.measurer.measure_circuit(circuit, 'Z'*n, shots)