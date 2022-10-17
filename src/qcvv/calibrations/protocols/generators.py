import numpy as np
from qibo import models, gates, get_backend
from utils import onequbit_clifford_params

class Generator():
    """
    Uniform Independent Random Sequence
    """

    def __init__(self, nqubits, **kwargs):
        """
        """
        self.nqubits = nqubits
        self.gate_generator = None
        self.invert = kwargs.get('invert', False)
        self.measurement = kwargs.get(
            'measurement', gates.M(*range(nqubits)))
    
    def minimal_representation(self):
        """
        """
        pass

    def build_gate(self):
        """
        """
        pass

    def __call__(self, sequence_length):
        """ For generating a sequence of circuits the object itself
        has to be called and the length of the sequence specified.

        Args:
            length (int) : How many circuits are created and put
                           together for the sequence.
        
        Returns:
        (list): with the minimal representation of the circuit.
        ``qibo.models.Circuit``: object which is executable as a
                                 simulation or on hardware
        """
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(self.nqubits)
        # Iterate over the sequence length.
        for _ in range(sequence_length):
            # Use the attribute to generate gates. This attribute can
            # differ for different classes since this encodes different
            # gat sets. For every loop retrieve the gate.
            gate = self.gate_generator()
            # Add the generated unitary gate to the list of circuits.
            circuit.add(gate)
        # For the standard randomized benchmarking scheme this is
        # useful but can also be ommitted and be done in the classical
        # postprocessing.
        if self.invert:
            # FIXME changed fusion gate calculation by hand since the
            # inbuilt function does not work.
            # Calculate the inversion matrix.
            inversion_unitary = circuit.invert().fuse().queue[0].matrix
            # Add it as a unitary gate to the circuit.
            circuit.add(gates.Unitary(inversion_unitary ,0))
        circuit.add(self.measurement)
        # No noise model added, for a simulation either the platform
        # introduces the errors or the error gates will be added
        # before execution.
        yield circuit

class GeneratorOnequbitcliffords(Generator):
    """
    TODO optimize the Clifford drawing
    """


    def __init__(self, nqubits, **kwargs):
        """
        """
        super().__init__(nqubits, **kwargs)
        # Overwrite the gate generator attribute from the motherclass with
        # the class specific generator.
        self.gate_generator = self.onequbit_clifford
    
    def minimal_representation(self, seed:int=None) -> tuple:
        """ One single Clifford qubit gate can be specified with four
        parameters.

        The parameters are hard coded and stored in the file utils.py as
        'onequbit_clifford_params', a tuple is drawn randomly.
        
        Args:
            seed (int): optional, set the starting seed for random
                        number generator.
        Returns:
            (tuple): tuple of 4 floats characterising one single
                      one qubit Clifford gate.
        """
        # Make it possible to set the seed for the random number generator.
        if seed is not None:
            # Get the backend and set the seed.
            backend = get_backend()
            backend.set_seed(seed)
        # Return the tuple with 4 numbers.
        # With the function self.
        return onequbit_clifford_params[
            np.random.randint(len(onequbit_clifford_params))]

    
    def build_gate(self, theta=0, nx=0, ny=0, nz=0):
        """ Four given parameters are used to build one Clifford gate.

        Args:
            theta (float) : An angle
            nx (float) : prefactor
            ny (float) : prefactor
            nz (float) : prefactor
        
        Returns:
            ``qibo.gates.Unitary`` with the drawn matrix as unitary.
        """
        matrix = np.array([[np.cos(theta/2) - 1.j*nz*np.sin(theta/2),
                        - ny*np.sin(theta/2) - 1.j*nx*np.sin(theta/2)],
                        [ny*np.sin(theta/2)- 1.j*nx*np.sin(theta/2),
                        np.cos(theta/2) + 1.j*nz*np.sin(theta/2)]])
        return gates.Unitary(matrix, 0)
    
    def onequbit_clifford(self, seed:int=None) -> tuple:
        """ Draws the parameters and builds the gate.

        Args:
            seed (int): optional, set the starting seed for random
                        number generator.
        Returns:
            random_parameters (tuple): 
            ``qibo.gates.Unitary``: the Clifford gate
        """
        random_parameters = self.minimal_representation(seed=seed)
        return self.build_gate(*random_parameters)