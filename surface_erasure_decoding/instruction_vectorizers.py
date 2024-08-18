from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional, Iterable
import stim
import numpy as np
from abc import ABC, abstractmethod
from itertools import cycle, islice

Etype_to_stim_target_fun = {
    'X': stim.target_x,
    'Y': stim.target_y,
    'Z': stim.target_z,
}

def chunked(iterable, chunk_size):
    """Yield successive n-sized chunks from the input iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def zip_longest_cycle(iterables):
    max_len = max(len(it) for it in iterables)
    cycled_iterables = [cycle(iterable) for iterable in iterables]
    for _ in range(max_len):
        yield tuple(next(it) for it in cycled_iterables)


@dataclass
class SQE: #stands for Single_qubit_error/event
    """
    Determines what's the type of error on the data qubit (I,X,Y,Z) and if it's heralded
    """
    type: str
    heralded: bool
    def __post_init__(self):
        assert self.type in ['I','X','Y','Z'] 

@dataclass
class MQE: #stands for Multi_qubit_error/event
    """
    Constitutes one of the many disjoint probabilities in an Error_mechanism
    """
    p: float
    list_of_SQE: List[SQE]
    def __post_init__(self):
        assert self.p >= 0, "can't create an event with below 0 probability?"

@dataclass
class InstructionVectorizer(ABC):
    list_of_MQE: List[MQE]

    def __post_init__(self):
        sum_of_prob = sum([mqe.p for mqe in self.list_of_MQE])
        assert  sum_of_prob > 1-1e-7 and sum_of_prob < 1+ 1e-7
        self.num_qubits = len(self.list_of_MQE[0].list_of_SQE) # the number of data qubits it describes
        assert all(len(mqe.list_of_SQE) == self.num_qubits for mqe in self.list_of_MQE) # Ensure the error model describes errors on a fixed amount of data qubits

    @abstractmethod
    def get_instruction(self, qubits:List[int]) -> List:
        pass   

    def __repr__(self):
        s = ''
        for mqe in self.list_of_MQE:
            s += str(mqe)
            s += '\n'
        return s

@dataclass
class InstructionVectorizerWithStepWiseProbs(InstructionVectorizer):
    def __post_init__(self):
        # Convert the probabilities to those used in CORRELATED_ERROR and ELSE_CORRELATED_ERRORs (used for non-vectorized methods only, which are Deprecated)
        self.list_of_MQE = sorted(self.list_of_MQE,reverse=True, key=lambda x: x.p)
        self.stepwise_probabilities = []
        prob_left = 1
        for event in self.list_of_MQE:
            stepwise_p = event.p / prob_left
            stepwise_p= max(min(stepwise_p, 1), 0)
            self.stepwise_probabilities.append(stepwise_p)
            prob_left -= event.p

@dataclass
class InstructionVectorizerWithPosteriorProbs(InstructionVectorizer):
    def __post_init__(self):
        # Prepare conditional probabilities for decoding erasure detection 
        # compute the arithmatic sum of I/X/Y/Z error rates over num_qubits qubits (using the sum without heralding in normal circuit is the same as using the *disjoint* components without heralding)
        #   also compute the sum of heralded pauli errors for num_qubits qubits
        self.Etype_to_sum = [None] * self.num_qubits
        self.Etype_to_heralded_sum = [None] * self.num_qubits
        self.conditional_probabilities = [None] * self.num_qubits
        self.p_herald = [0] * self.num_qubits
        for i in range(self.num_qubits):
            self.Etype_to_sum[i] = {
                'I':0,
                'X':0,
                'Y':0,
                'Z':0
            }
            self.Etype_to_heralded_sum[i] = {
                'I':0,
                'X':0,
                'Y':0,
                'Z':0
            }
            for mqe in self.list_of_MQE:
                Etype_on_this_qubit = mqe.list_of_SQE[i].type
                self.Etype_to_sum[i][Etype_on_this_qubit] += mqe.p
                if mqe.list_of_SQE[i].heralded == True:
                    self.Etype_to_heralded_sum[i][Etype_on_this_qubit] += mqe.p

            # Compute the conditional probabilities
            self.p_herald[i] = sum(self.Etype_to_heralded_sum[i].values())
            if self.p_herald[i] != 0:
                self.conditional_probabilities[i] = [
                    self.Etype_to_heralded_sum[i]['X']/self.p_herald[i], 
                    self.Etype_to_heralded_sum[i]['Y']/self.p_herald[i], 
                    self.Etype_to_heralded_sum[i]['Z']/self.p_herald[i],
                    (self.Etype_to_sum[i]['X']-self.Etype_to_heralded_sum[i]['X'])/(1-self.p_herald[i]),
                    (self.Etype_to_sum[i]['Y']-self.Etype_to_heralded_sum[i]['Y'])/(1-self.p_herald[i]),
                    (self.Etype_to_sum[i]['Z']-self.Etype_to_heralded_sum[i]['Z'])/(1-self.p_herald[i])
                ]
        self.heralded_locations_one_hot_encoding  = [prob > 0 for prob in self.p_herald]
        self.herald_locations = np.where(self.heralded_locations_one_hot_encoding)[0]
        self.num_herald_locations = np.sum(self.heralded_locations_one_hot_encoding)

    def get_new_ancillas_array_update_list(self,data_qubits_array,index_in_list):
        num_parallel = data_qubits_array.shape[1]
        num_ancillas= num_parallel * self.num_herald_locations
        
        ancillas = np.zeros(data_qubits_array.shape, dtype=int)
        counter = index_in_list[0]
        fill_values = np.arange(counter, counter + num_ancillas)
        fill_values = fill_values.reshape(num_parallel, self.num_herald_locations).T
        ancillas[self.herald_locations, :] = fill_values
        index_in_list[0] += num_ancillas
        return ancillas, num_parallel
    
@dataclass
class NormalInstructionVectorizer(InstructionVectorizerWithStepWiseProbs):
    instruction_name: Optional[str] = field(default=None)
    instruction_arg: Union[float, List[float]] = field(default=None)
    vectorizable: Optional[bool] = field(default=None)
    
    def __post_init__(self):
        InstructionVectorizer.__post_init__(self)
        if  self.instruction_name == None or self.instruction_arg == None:
            self.vectorizable = False
            InstructionVectorizerWithStepWiseProbs.__post_init__(self)
        else:
            self.vectorizable = True

    def get_instruction(self, qubits:List[int]) -> List:
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        if self.vectorizable:
            list_of_args = []
            list_of_args.append([self.instruction_name,qubits,self.instruction_arg])
            return list_of_args
        else:
            list_of_args_gathered = []
            for sets_of_qubits in chunked(iterable=qubits, chunk_size = self.num_qubits):
                list_of_args = []
                for mqe, stepwise_p in zip(self.list_of_MQE,self.stepwise_probabilities):
                    targets = []
                    for sqe,qubit in zip(mqe.list_of_SQE, sets_of_qubits):# the len of qubits can be smaller than len of event.list_of_Etype, but the smallest len count will determin how many iterations are run in zip()
                        if sqe.type != 'I':
                            targets.append(Etype_to_stim_target_fun[sqe.type](qubit))
                    list_of_args.append(["ELSE_CORRELATED_ERROR",targets,stepwise_p])
                list_of_args[0][0] = "CORRELATED_ERROR"
                list_of_args_gathered.extend(list_of_args)
            return list_of_args_gathered   

@dataclass
class ErasureInstructionVectorizer(InstructionVectorizerWithStepWiseProbs, InstructionVectorizerWithPosteriorProbs):
    '''
        Whether an erasure insturction is vectorizable is dependent on whether there's correlation between two data qubits.
        If during a 2-qubit gate, the two qubits are independently erased, then we apply two pairs of PAULI_CHANNEL_2.
        But if the 2-qubit gate is described by some correlated erasure error, as in PHYSICAL REVIEW X 13, 041013 (2023), 
            then the mechanism involving more than 2 qubits need to be modeled by CORRELATED_ERROR, and CORRELATED_ERROR are not vectorizable.

        instruction_name and instruction_arg can be used to describe one PAULI_CHANNEL_2 that is applied to one (pair (with broadcasting)) of qubits
            or two PAULI_CHANNEL_2 that is separatly applied to a pair of qubits
    '''
    instruction_name: Optional[Union[str, List[float]]] = field(default=None)
    instruction_arg: Optional[Union[float, List[float],List[List[float]]]] = field(default=None)
    vectorizable: Optional[bool] = field(default=None)
    def __post_init__(self):
        InstructionVectorizer.__post_init__(self)
        if self.instruction_name == None or self.instruction_arg == None:
            self.vectorizable = False
            InstructionVectorizerWithStepWiseProbs.__post_init__(self)
        else:
            self.vectorizable = True
            if isinstance(self.instruction_name,List):
                assert len(self.instruction_name) == len(self.instruction_arg)
            else:
                self.instruction_name = [self.instruction_name]
                self.instruction_arg = [self.instruction_arg]
            InstructionVectorizerWithPosteriorProbs.__post_init__(self)

    def get_instruction(self, qubits:List[int],next_ancilla_qubit_index_in_list:List[int]) -> List:
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        data_qubits_array = np.array(qubits).reshape(-1,self.num_qubits).T
        ancillas, num_parallel = self.get_new_ancillas_array_update_list(data_qubits_array,next_ancilla_qubit_index_in_list)

        if self.vectorizable:
            list_of_args = []
            for data_qubits_batch,ancilla_batch, instruction, arg in zip_longest_cycle([data_qubits_array,ancillas, self.instruction_name, self.instruction_arg]):
                interposed_array = np.empty(num_parallel*2, dtype=int)
                interposed_array[::2] = data_qubits_batch
                interposed_array[1::2] = ancilla_batch
                list_of_args.append([instruction, interposed_array, arg])
            return list_of_args
        else:
            ancillas = ancillas.T.flatten()
            ancilla_idx = 0
            list_of_args_gathered = []
            for sets_of_qubits in chunked(iterable=qubits, chunk_size = self.num_qubits):
                list_of_args = []
                for mqe, stepwise_p in zip(self.list_of_MQE,self.stepwise_probabilities):
                    targets = []
                    for sqe,circuit_qubit_idx in zip(mqe.list_of_SQE, sets_of_qubits):
                        if sqe.type != 'I':
                            targets.append(Etype_to_stim_target_fun[sqe.type](circuit_qubit_idx))
                        if sqe.heralded:# If it heralded, then the corresponding ancilla must have been assigned on this qubit for this Gate_error_model 
                            targets.append(Etype_to_stim_target_fun['X'](ancillas[ancilla_idx]))
                            ancilla_idx += 1
                    list_of_args.append(["ELSE_CORRELATED_ERROR",targets,stepwise_p])
                list_of_args[0][0] = "CORRELATED_ERROR" # The first instruction should be CORRELATED_ERROR in stim
                list_of_args_gathered.extend(list_of_args)
            return list_of_args_gathered   

@dataclass
class DynamicInstructionVectorizer(InstructionVectorizerWithPosteriorProbs):
    def __post_init__(self):
        InstructionVectorizer.__post_init__(self)
        InstructionVectorizerWithPosteriorProbs.__post_init__(self)

    def get_instruction(self,qubits:List[int],
                        erasure_measurement_index_in_list:List[int],
                        single_measurement_sample:Union[List,np.array]) -> List:
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        data_qubits_array = np.array(qubits).reshape(-1,self.num_qubits).T

        if self.num_herald_locations > 0:
            ancillas,num_parallel = self.get_new_ancillas_array_update_list(data_qubits_array,erasure_measurement_index_in_list)
            
            # Get bool array signifing erasure detection
            erasure_meas = np.zeros(ancillas.shape, dtype=bool)
            erasure_meas[self.herald_locations, :] = single_measurement_sample[ancillas[self.herald_locations, :]]
            
            # For each qubit location, get two conditional probabilities array or the static probability if erasure conversion not used. 
            list_of_args = []
            for i in range(self.num_qubits): # This looks like a loop, but this loop is at most length-2. It's still parrallized
                if i in self.herald_locations: 
                    converted_data_qubits = data_qubits_array[i][np.where(erasure_meas[i])[0]]
                    no_detection_data_qubits = data_qubits_array[i][np.where(~erasure_meas[i])[0]]
                    list_of_args.append(["PAULI_CHANNEL_1", converted_data_qubits, [self.conditional_probabilities[i][0], self.conditional_probabilities[i][1], self.conditional_probabilities[i][2]]])
                    list_of_args.append(["PAULI_CHANNEL_1", no_detection_data_qubits, [self.conditional_probabilities[i][3], self.conditional_probabilities[i][4], self.conditional_probabilities[i][5]]])
                else:
                    list_of_args.append(["PAULI_CHANNEL_1", data_qubits_array[i], [self.Etype_to_sum[i]['X'], self.Etype_to_sum[i]['Y'], self.Etype_to_sum[i]['Z']]])
        else:
            list_of_args = []
            for i in range(self.num_qubits):
                list_of_args.append(["PAULI_CHANNEL_1", data_qubits_array[i], [self.Etype_to_sum[i]['X'], self.Etype_to_sum[i]['Y'], self.Etype_to_sum[i]['Z']]])
        return list_of_args   
    
@dataclass
class DeterministicInstructionVectorizer(InstructionVectorizer):
    def get_instruction(self, qubits: Union[List[int], Tuple[int]]) -> List:
        pass   
