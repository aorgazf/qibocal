
from qcvv.data import Data


class Experiment():
    """
    """
    def __init__(self, sequence_lengths:list, qubits:list,
            nshots:int=1024, circuit_generator:str=None,
            inverse=False, **kwargs) -> None:
        self.circuit_generator = circuit_generator
        self.sequence_lengths = sequence_lengths
        self.qubits = qubits
        self.nshots = nshots
        self.inverse = inverse

    def prebuild_a_save(self, **kwargs):
        """ 
        """
        for ll in self.sequence_lengths:
            kennzeichnung = next(self.circuit_generator)
        # experiment_data = Data("experiment",
        #     quantities=list(sequence_lengths))
        # yielda experiment_data
        # sequencel_data
        pass

    def retrive_from_file(self, filename:str, **kwargs):
        """
        """
        pass

    def build_onthefly(self, **kwargs):
        """
        """
        pass

    def execute_a_measure(self, **kwargs):
        """
        """
        pass
    
    def convert_to_object(generator_str):
        """
        """
        pass
