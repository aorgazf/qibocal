from ctypes import Union
from qcvv.calibrations.protocols.utils import dict_to_txt
from qcvv.calibrations.protocols.utils import pkl_to_list
from qcvv.data import Data
from qibo.noise import PauliError, NoiseModel
from qibo import gates
import numpy as np
from os.path import isdir
from os.path import isfile
from os import mkdir
import pdb
from  qcvv.calibrations.protocols.generators import *
from typing import Union
from itertools import product


class Experiment():
    """  The experiment class has methods to build, save, load and execute
    circuits with different depth with a given generator for random circuits. 
    After executing the experiment the outcomes are stored too.


    Attributes:
        TODO work in possible error models better

    circuits_list (list) : The list of lists of circuits. 
            axis 1: different runs, axis 2: different sequence lengths.

    """
    def __init__(self, circuit_generator:Generator=None,
            sequence_lengths:list=None, qubits:list=None, runs:int=None,
            nshots:int=1024, **kwargs) -> None:
        self.circuit_generator = circuit_generator
        self.sequence_lengths = sequence_lengths
        self.qubits = qubits
        self.runs = runs
        self.nshots = nshots
        if hasattr(circuit_generator, 'invert'):
            self.inverse = circuit_generator.invert
    
    ############################### PROPERTIES ###############################

    @property
    def data_circuits(self):
        """
        """
        # Initiate the data structure from qibocal.
        data_circs = Data(
            'circuits', quantities=list(self.sequence_lengths))
        # Store the data in a pandas dataframe. The columns are indexed by the
        # different sequence lengths. The rows are indexing the different runs.
        for count in range(self.runs):
            # The data object takes dictionaries.
            data_circs.add({
                self.sequence_lengths[i]:self.circuits_list[count][i] \
                for i in range(len(self.sequence_lengths))})
        return data_circs

    @property
    def data_samples(self):
        """
        """
        # Initiate the data structure where the outcomes will be stored.
        data_samples = Data(
            'samples', quantities=list(self.sequence_lengths))
        # Go through every run and this way store the whole list of samples.
        for count in range (self.runs):
            # Add the samples to the data structure.
            data_samples.add(
                {self.sequence_lengths[i]:self.outcome_samples[count][i] \
                    for i in range(len(self.sequence_lengths))})
        return data_samples

    @property
    def data_probabilities(self):
        """
        """
        # Initiate the data structure where the outcomes will be stored.
        data_probs = Data(
            'probabilities', quantities=list(self.sequence_lengths))
        # Go through every run and this way store the whole list
        # of probabilites.
        for count in range (self.runs):
            # Put the probabilities into the data object.
            data_probs.add(
                {self.sequence_lengths[i]:self.outcome_probabilities[count][i] \
                for i in range(len(self.sequence_lengths))})
        return data_probs
    
    ############################## CLASS METHODS ##############################

    @classmethod
    def retrieve_experiment(cls, path:str, **kwargs):
        """
        """
        from qcvv.calibrations.protocols.utils import dict_from_comments_txt
        # Initiate an instance of the experiment class.
        obj = cls()
        # Get the metadata in form of a dictionary.
        metadata_dict = dict_from_comments_txt(f'{path}metadata.txt')
        # The circuit generator has to be restored, this will get the class.
        Generator = eval(metadata_dict['circuit_generator'])
        # Build the generator.
        circuit_generator = Generator(metadata_dict['qubits'])
        # Write it to the dictionary.
        metadata_dict['circuit_generator'] = circuit_generator
        # Give the objects the attributes as a dictionary. Every attribute
        # would be overwritten by that.
        obj.__dict__ = metadata_dict
        # Store the diven path.
        obj.directory = path
        # Get the circuits list and make it an attribute.
        obj.load_circuits(path)
        # Try to load the outcomes. 
        try:
            obj.load_samples(path)
            obj.load_probabilities(path)
        except FileNotFoundError:
            # If there are no outcomes (yet), there will be no files.
            print('No outcomes to retrieve.')
        return obj

    ################################# METHODS #################################

    ############################## Build ##############################

    def build(self, **kwargs):
        """ Build a list out of the circuits required to run for the wanted
        experiment.
        """
        # Use the __call__ function of the circuit generator to retrieve a
        # random circuit 'runs' many times for each sequence length.
        circuits_list = [
            [next(self.circuit_generator(length))
            for length in self.sequence_lengths] for _ in range(self.runs)]
        # Create an attribute.
        # TODO should that be necessary if the experiment is stored in a 
        # pikle file?
        self.circuits_list = circuits_list
        return circuits_list
    
    def build_onthefly(self, **kwargs):
        """
        """
        pass

    def build_a_save(self, **kwargs):
        """ 
        """
        # Build the whole list of circuits.
        self.build(**kwargs)
        # Store the list of circuits.
        self.save_experiment(**kwargs)
    
    def build_noise(self, **kwargs):
        """
        """
        pass
    
    ################################ Execute ################################

    def execute_experiment(self, **kwargs):
        """ FIXME the circuits have to be build already (or loaded), 
        add something to check that and if they were not build yet build or
        load them.

        Args:
            kwargs (dict):
                'paulierror_noiseparams' = [p1, p2, p3]
        """
        # Initiate the outcome lists, one for the single shot samples and
        # one for the probabilities.
        self.outcome_samples, self.outcome_probabilities = [], []
        # If the circuits are simulated and not run on quantum hardware, the
        # noise has to be simulated, too.
        if kwargs.get('paulierror_noiseparams'):
            # Insert artificial noise, namely random Pauli flips.
            pauli = PauliError(*kwargs.get('paulierror_noiseparams'))
            noise = NoiseModel()
            # The noise should be applied with each unitary in the circuit.
            noise.add(pauli, gates.Unitary)
        # Makes code easier to read.
        amount_m = len(self.sequence_lengths)
        # Loop 'runs' many times over the whole protocol.
        for count_runs in range(self.runs):
            # Initiate two lists to store the outcome for every sequence.
            probs_list, samples_list = [], []
            # Go through every sequence in the protocol.
            for count_m in range(amount_m):
                # Get the circuit.
                circuit = self.circuits_list[count_runs][count_m]
                # For the simulation the noise has to be added to the circuit.
                if kwargs.get('paulierror_noiseparams'):
                    # Add the noise to the circuit (more like the other way
                    # around, the circuit to the noise).
                    noisy_circuit = noise.apply(circuit)
                    # Execute the noisy circuit.
                    executed = noisy_circuit(nshots=self.nshots)
                else:
                    # Execute the qibo circuit without artificial noise.
                    executed = circuit(nshots=self.nshots)
                # FIXME The samples (zeros and ones per shot) acquisition does 
                # not work for quantum hardware yet.
                try:
                    # Get the samples from the executed gate. It should be an
                    # object filled with as many integers as used shots.
                    # Append the samples.
                    samples_list.append(executed.samples())
                except:
                    print('Retrieving samples not possible.')
                    # pass
                # Either way store the probabilities. Since
                # 'executed.probabilities()' only contains an entry for qubit
                # if it is nonzero, the shape can vary, fix that FIXME.
                # Store them.
                probs_list.append(list(executed.probabilities()))
            # For each run store the temporary lists in the attribute.
            # It could happend that the samples list is empty if the samples
            # cannot be retrieved.
            self.outcome_samples.append(samples_list)
            self.outcome_probabilities.append(probs_list)

    def execute_a_save(self, **kwargs):
        """
        """
        self.execute_experiment(**kwargs)
        self.save_outcome(**kwargs)

    ###################### Datastructures and save/load ######################
    
    def make_directory(self, **kwargs):
        """ Make the directory where the experiment will be stored.
        """
        from datetime import datetime
        overall_dir = 'experiments/'
        # Check if the overall directory exists. If not create it.
        if not isdir(overall_dir):
            mkdir(overall_dir)
        # Get the current date and time.
        dt_string = datetime.now().strftime("%y%b%d_%H%M%S")
        # Get the name of the generator.
        gname = self.circuit_generator.__class__.__name__
        # Every generator for the circuits gets its own directory.
        directory_generator = f'{overall_dir}{gname}/'
        if not isdir(directory_generator):
            mkdir(directory_generator)
        # Name the final directory for this experiment.
        directory = f'{directory_generator}experiment{dt_string}/'
        if not isdir(directory):
            mkdir(directory)
        # Store this as an attribute.
        self.directory = directory
        return directory

    def save_circuits(self, **kwargs) -> None:
        """ Save the given circuits list. 
        FIXME if the circuits were executed already this does not work!! 

        Args:
            kwargs (dict)
        
        Returns:
            None
        """
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, 'directory'):
            # Make and get the directory.
            self.make_directory()
        # Use the property.
        data_circs = self.data_circuits 
        # Save the circuits in pickle format.
        data_circs.to_pickle(self.directory)

    def save_metadata(self, **kwargs):
        """
        """
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, 'directory'):
            # Make and get the directory.
            self.make_directory()
        # Store the metadata in a .txt file. For that create a dictionary.
        # Store any parameters given through kwargs.
        metadata_dict = {
            'qubits' : self.qubits,
            'nshots' : self.nshots,
            'runs' : self.runs,
            'inverse' : self.inverse,
            'circuit_generator' : self.circuit_generator.__class__.__name__
        }
        # One file in the directory stores the meta data.
        metadata_filename = f'{self.directory}metadata.txt'
        # Write the meta data as comments to the .txt file.
        dict_to_txt(metadata_filename, metadata_dict, openingstring='w')
        # The file is automatically closed.
    
    def save_experiment(self, **kwargs):
        """
        """
        self.save_metadata(**kwargs)
        self.save_circuits(**kwargs)
        return self.directory

    def save_outcome(self, **kwargs):
        """
        """
         # Check if there has been made a directory already for this experiment.
        if not hasattr(self, 'directory'):
            # Make and get the directory.
            self.make_directory()
        if isfile(f'{self.directory}metadata.txt'):
            dict_to_txt(f'{self.directory}metadata.txt',
            kwargs, comments=True, openingstring='a')
        # Use the properties.
        data_probs = self.data_probabilities
        data_samples = self.data_samples
        # Save the data structures.
        data_samples.to_pickle(self.directory)
        data_probs.to_pickle(self.directory)

    def load_circuits(self, path:str, **kwargs):
        """
        """
        if isfile(f'{path}circuits.pkl'):
            # Get the pandas data frame from the pikle file.
            sequences_frompkl, circuits_list = pkl_to_list(
                f'{path}circuits.pkl')
            # Check if the attribute does not exist yet.
            if not hasattr(self, 'sequence_lengths'):
                # The pickeling process reverses the order, reoder ot.
                self.sequence_lengths = np.array(sequences_frompkl)[::-1]
            # Store the outcome as an attribute to further work with its.
            self.circuits_list = [x[::-1] for x in circuits_list]
            return self.circuits_list
        else:
            raise FileNotFoundError('There is no file for circuits.')

    def load_samples(self, path:str, **kwargs):
        """
        """
        if isfile(f'{path}samples.pkl'):
            # Get the pandas data frame from the pikle file.
            sequences_frompkl, samples_list = pkl_to_list(f'{path}samples.pkl')
            # Make sure that the order is the same.
            assert np.array_equal(
                np.array(sequences_frompkl)[::-1], self.sequence_lengths), \
                'The order of the restored outcome is not the same as when build'
            # Store the outcome as an attribute to further work with its.
            self.outcome_samples = [x[::-1] for x in samples_list]
            return self.outcome_samples
        else:
            raise FileNotFoundError('There is no file for samples.')
    
    def load_probabilities(self, path:str, **kwargs):
        """
        """
        if isfile(f'{path}probabilities.pkl'):
            # Get the pandas data frame from the pikle file.
            sequences_frompkl, probabilities_list = pkl_to_list(
                f'{path}probabilities.pkl')
            # Make sure that the order is the same, right now the
            # order is reversed.
            assert np.array_equal(
                np.array(sequences_frompkl)[::-1], self.sequence_lengths), \
                'The order of the restored outcome is not the same as when build'
            # Store the outcome as an attribute to further work with its.
            self.outcome_probabilities = [x[::-1] for x in probabilities_list]
            return self.outcome_probabilities
        else:
            raise FileNotFoundError('There is no file for probabilities.')

    ########################### Outcome processing ###########################

    def probabilities(
            self, averaged:bool=True, run:Union[int, list]=None,
            from_samples:bool=True, **kwargs) -> np.ndarray:
        """
        """
        # Check if the samples attribute is not empty e.g. the first entry is
        # not just an empty list.
        if len(self.outcome_samples[0]) != 0 and from_samples:
            # Create all possible state vectors.
            allstates = np.array(list(product([0,1], repeat=len(self.qubits))))
            # The attribute should be lists out of lists out of lists out
            # of lists, make it an array.
            samples = np.array(self.outcome_samples)
            if averaged:
                # Put the runs together, now the shape is
                # (amount sequences, runs*nshots, qubits).
                samples_conc = np.concatenate(samples, axis=1)
                # For each sequence length count the different state vectors and
                # divide by the total number of shots.
                probs = [[np.sum(np.product(samples_conc[countm]==state, axis=1))  
                    for state in allstates]
                    for countm in range(len(self.sequence_lengths))]
                probs = np.array(probs)/(self.runs*self.nshots)
            else:
                # If only a specific run (runs) is requested, choose that one.
                if run:
                    # Since the concatination in the next step only works when
                    # there are 4 dimensions, reshape it to 4 dimensions.
                    samples = samples[run].reshape(
                        -1, len(self.sequence_lengths),
                        self.nshots, len(self.qubits))
                # Do the same thing as above just for every run.
                probs = [[[
                    np.sum(np.product(samples[countrun, countm]==state, axis=1))  
                    for state in allstates]
                    for countm in range(len(self.sequence_lengths))]
                    for countrun in range(len(samples))]
                probs = np.array(probs)/(self.nshots)
        else:
            # The actual probabilites are used.
            probs = np.array(self.outcome_probabilities)
            if averaged:
                # If needed, average over the different runs for each sequence
                # length.
                probs = np.average(probs, axis=0)
            # Or pick a run. But if averaged is set to True this will not
            # happen.
            if run:
                probs = probs[run]
        return probs

    def samples(self, run:Union[int, list]=None) -> np.ndarray:
        """
        """
        # Check if the samples attribute is not empty e.g. the first entry is
        # not just an empty list.
        if len(self.outcome_samples[0]) != 0:
            # It should be lists out of lists out of lists out of lists,
            # make it an array.
            samples =  np.array(self.outcome_samples)
            if run:
                # Specific runs can be chosen.
                samples = samples[run]
            return samples
        else:
            raise ValueError('No samples there. Try probabilities().')

    def postprocess(self, **kwargs):
        """
        """
        pass
