from dataclasses import dataclass
from typing import List, Union, Tuple, Optional
import stim
import numpy as np

class Ancilla_tracker:
    def __init__(self, first_ancilla_qubit_index) -> None:
        self.next_ancilla_qubit_index = first_ancilla_qubit_index
        self.bare_list_of_ancillas = []
        self.ancilla_to_xyz = []

    def assign_ancilla(self, x: float, y: float, z: float):
        self.bare_list_of_ancillas.append(self.next_ancilla_qubit_index)
        self.ancilla_to_xyz.append([self.next_ancilla_qubit_index, x, y, z])
        self.next_ancilla_qubit_index += 1
        return self.next_ancilla_qubit_index - 1

Etype_to_stim_target_fun = {
    'X': stim.target_x,
    'Y': stim.target_y,
    'Z': stim.target_z,
}

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
        assert self.p > 0, "can't create an event with 0 probability?"

    

@dataclass
class Error_mechanism:
    """
    A Gate_error_mechanism constitutes multiple MQEs, can describe the intrinsic error or the erasure conversion of the gate.
    This class specify some error events with *disjoint* probabilities. 
    The MQEs are disjoint because it's good enough approximation to neglect events where multiple MQEs happening at the same time.
    It also specify which error events can be heralded, and herald rate.

    When used in gen_erasure_conversion_circuit, the error model is fully implemented
    When used in gen_normal_circuit, the error model is used without implementing heralding (operator on the last (or last two) relative qubit index is neglected)
    When used in gen_dynamic_circuit, the contidional arithmatically summed probabilities are used, and without implementing heralding

    It is made up of a continuous chunk of CORRELATED_ERROR and ELSE_CORRELATED_ERRORs. 
    The parameters used in CORRELATED_ERROR and ELSE_CORRELATED_ERRORs are different from what's seen here. See documentation of stim.
    """
    list_of_MQE: Optional[List[MQE]]
    ancilla_tracker_instance: Optional[Ancilla_tracker] = None
    erasure_measurement_index_in_list: Optional[int] = None
    single_measurement_sample:  Optional[Union[List,np.array]] = None
    _herald_check_cache: bool = None
    def __post_init__(self):
  
        sum_of_prob = sum([mqe.p for mqe in self.list_of_MQE])
        assert  sum_of_prob > 1-1e-7 and sum_of_prob < 1+ 1e-7

        self.num_qubits = len(self.list_of_MQE[0].list_of_SQE) # the number of data qubits it describes
        assert all(len(mqe.list_of_SQE) == self.num_qubits for mqe in self.list_of_MQE) # Ensure the error model describes errors on a fixed amount of data qubits

        self.list_of_MQE = sorted(self.list_of_MQE,reverse=True, key=lambda x: x.p)
        # convert the probabilities to those used in CORRELATED_ERROR and ELSE_CORRELATED_ERRORs
        self.stepwise_probabilities = []
        prob_left = 1
        for event in self.list_of_MQE:
            stepwise_p = event.p / prob_left
            stepwise_p= max(min(stepwise_p, 1), 0)
            self.stepwise_probabilities.append(stepwise_p)
            prob_left -= event.p
            
        # compute the arithmatic sum of I/X/Y/Z error rates over num_qubits qubits (using the sum without heralding in normal circuit is the same as using the *disjoint* components without heralding)
        #   also compute the sum of heralded pauli errors for num_qubits qubits
        self.Etype_to_sum = [None] * self.num_qubits
        self.Etype_to_heralded_sum = [None] * self.num_qubits
        self.conditional_probabilities = [None] * self.num_qubits
        self.p_herald = [0] * self.num_qubits
        self.trivial_when_not_detected = [False] * self.num_qubits
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
                if self.conditional_probabilities[i][3] == 0 and self.conditional_probabilities[i][4] == 0 and self.conditional_probabilities[i][5] == 0:
                    self.trivial_when_not_detected[i] = True
        self.heralded_locations_one_hot_encoding  = [prob > 0 for prob in self.p_herald]
        self.herald_locations = np.where(self.heralded_locations_one_hot_encoding)[0]
        self.num_herald_locations = np.sum(self.heralded_locations_one_hot_encoding)
        if sum(self.p_herald) == 0:
            self.is_erasure = False
        else:
            self.is_erasure = True

    def get_instruction(self, 
                        qubits: Union[List[int], Tuple[int]],
                        mode:str):
        '''
        return list of args that can be used in  stim.circuit.append()
        '''
        assert self.num_qubits == len(qubits), f"supposed to take in {self.num_qubits} qubits, but took in {qubits}"

        if mode == 'normal':
            list_of_args = []
            for mqe, stepwise_p in zip(self.list_of_MQE,self.stepwise_probabilities):
                targets = []
                for sqe,qubit in zip(mqe.list_of_SQE, qubits):# the len of qubits can be smaller than len of event.list_of_Etype, but the smallest len count will determin how many iterations are run in zip()
                    if sqe.type != 'I':
                        targets.append(Etype_to_stim_target_fun[sqe.type](qubit))
                list_of_args.append(["ELSE_CORRELATED_ERROR",targets,stepwise_p])
            list_of_args[0][0] = "CORRELATED_ERROR"
            return list_of_args
        elif mode == 'erasure':
            assert type(self.ancilla_tracker_instance) == Ancilla_tracker, "ancilla tracker not assigned"
            
            ancillas = []
            for i in range(len(qubits)):
                if self.p_herald[i] > 0:
                    ancillas.append(self.ancilla_tracker_instance.assign_ancilla(self.Etype_to_sum[i]['X'],self.Etype_to_sum[i]['Y'],self.Etype_to_sum[i]['Z']))
                else: # Attention: in erasure mode it does not assign ancilla to qubit unless that qubit has possitive herald probability
                    ancillas.append(None)
            list_of_args = []
            for mqe, stepwise_p in zip(self.list_of_MQE,self.stepwise_probabilities):
                targets = []
                for sqe,circuit_qubit_idx,ancilla in zip(mqe.list_of_SQE, qubits,ancillas):
                    if sqe.type != 'I':
                        targets.append(Etype_to_stim_target_fun[sqe.type](circuit_qubit_idx))
                    if sqe.heralded:# If it heralded, then the corresponding ancilla must have been assigned on this qubit for this Gate_error_model 
                        targets.append(Etype_to_stim_target_fun['X'](ancilla))
                list_of_args.append(["ELSE_CORRELATED_ERROR",targets,stepwise_p])
            list_of_args[0][0] = "CORRELATED_ERROR" # The first instruction should be CORRELATED_ERROR in stim
            return list_of_args
        elif mode == 'dummy_erasure':
            # Here I use PAULI_CHANNEL_TWO to make it easier to convert erasure location to 
            assert type(self.ancilla_tracker_instance) == Ancilla_tracker, "ancilla tracker not assigned"
            
            list_of_args = []
            ancillas = []
            for i in range(len(qubits)):
                if self.p_herald[i] > 0:
                    ancillas.append(self.ancilla_tracker_instance.assign_ancilla(self.Etype_to_sum[i]['X'],self.Etype_to_sum[i]['Y'],self.Etype_to_sum[i]['Z']))
                    list_of_args.append(["PAULI_CHANNEL_2",[qubits[i], ancillas[-1]],[
                        # ix iy iz
                        0, 0, 0,
                        # xi xx xy xz
                        0.1, 0.1, 0, 0,
                        # yi yx yy yz
                        0, 0, 0, 0,
                        # zi zx zy zz
                        0, 0, 0, 0
                    ]])
                else: # Attention: in erasure mode it does not assign ancilla to qubit unless that qubit has possitive herald probability
                    ancillas.append(None)
            return list_of_args
        elif mode == 'dynamic':
            # This mode is only used for decoding, not for sampling
            # In this mode, all X/Y/Z errors on data qubits are considered independent
            # so I use PAULI_CHANNEL_1 to append the conditional, 1-q errors
            # Each PAULI_CHANNEL_1's X/Y/Z are supposed to be disjoint, but approximated as independent in DEM, but it's good approximation

            # First determine if none of the sites used erasure detection,  fall back to normal mode in that case
            if self._herald_check_cache is None:
                self._herald_check_cache = not any(prob > 0 for prob in self.p_herald)
            if self._herald_check_cache:
                return self.get_instruction(qubits,mode='normal')
            
            # If there's ancilla assigned
            ancilla_meas_idxs = []
            for i in range(len(qubits)):
                if self.p_herald[i] > 0:
                    ancilla_meas_idxs.append(self.erasure_measurement_index_in_list[0])
                    self.erasure_measurement_index_in_list[0] += 1
                else:
                    ancilla_meas_idxs.append(None)
            list_of_args = []
            for i, qubit in enumerate(qubits):
                # First determine if we assign(ed) ancilla to the qubit or not
                if ancilla_meas_idxs[i] != None:
                    meas = self.single_measurement_sample[ancilla_meas_idxs[i]]
                    if meas == 0 and not self.trivial_when_not_detected[i]: # No erasure
                        list_of_args.append(["PAULI_CHANNEL_1", qubit, [self.conditional_probabilities[i][3], self.conditional_probabilities[i][4], self.conditional_probabilities[i][5]]])
                    else :
                        list_of_args.append(["PAULI_CHANNEL_1", qubit, [self.conditional_probabilities[i][0], self.conditional_probabilities[i][1], self.conditional_probabilities[i][2]]])
                else:# We didn't use erasure detection on this site, fall back to static probabilities for this particular site (indistinction to normal mode)
                    list_of_args.append(["PAULI_CHANNEL_1", qubit, [self.Etype_to_sum[i]['X'], self.Etype_to_sum[i]['Y'], self.Etype_to_sum[i]['Z']]])
            return list_of_args
        else:
            raise Exception("unsupported mode")
    
    def get_dynamic_instruction_vectorized(self,
                                           qubits:List[int]):
        """
        Accept qubits in the style of stim instructions, len(qubits) == integer multiple of self.num_qubits
        """
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        data_qubits_array = np.array(qubits).reshape(self.num_qubits,-1)
        # Each column of data_qubits_array has length num_qubits
        # It's like 
        #  array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        if self.num_herald_locations > 0:
            num_ops = data_qubits_array.shape[1]
            num_ancillas= num_ops * self.num_herald_locations
            # Assign ancilla array based on self.p_herald[i] > 0?, add the number of ancilla to self.erasure_measurement_index_in_list[0]
            ancilla = np.zeros(data_qubits_array.shape, dtype=int)
            counter = self.erasure_measurement_index_in_list[0]
            fill_values = np.arange(counter, counter + num_ancillas)
            # TODO: Somehow the line below leads to incorrect erasure measurement index. 
            # I think the original line makes sense but it doesn't work
            # fill_values = fill_values.reshape(num_ops, self.num_herald_locations).T # original
            fill_values = fill_values.reshape( self.num_herald_locations,num_ops) # What works
            ancilla[self.herald_locations, :] = fill_values
            self.erasure_measurement_index_in_list[0] += num_ancillas
            
            # Get bool array signifing erasure detection
            erasure_meas = np.zeros(ancilla.shape, dtype=bool)
            erasure_meas[self.herald_locations, :] = self.single_measurement_sample[ancilla[self.herald_locations, :]]
            
            # For each qubit location, get two conditional probabilities array or the static probability if erasure conversion not used. 
            list_of_args = []
            for i in range(self.num_qubits):
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



    def __repr__(self):
        s = ''
        for mqe in self.list_of_MQE:
            s += str(mqe)
            s += '\n'
        return s
@dataclass
class Gate_error_model:
    """
    A Gate_error_model contains one or more Gate_error_mechanisms,
    Different Gate_error_mechanisms within a Gate_error_model are considered independent (they can happen at the same time at higher order probability)

    A Gate_error_model is used to describe erasure conversion and normal error mechannism are independent,
    A Gate_error_model can only have one erasure conversion mechanism.
    """

    list_of_mechanisms: List[Error_mechanism]

    def __post_init__(self):
        if len(self.list_of_mechanisms) == 0:
            self.trivial = True
            return
        else:
            self.trivial = False
            
        assert all([mechanism.num_qubits == self.list_of_mechanisms[0].num_qubits for mechanism in self.list_of_mechanisms])
        assert sum([mechanism.is_erasure for mechanism in self.list_of_mechanisms]) <= 1
    def set_ancilla_tracker_instance(self,ancilla_tracker_instance: Ancilla_tracker):
        for mechanism in self.list_of_mechanisms:
            mechanism.ancilla_tracker_instance = ancilla_tracker_instance
    def set_erasure_measurement_index_in_list(self,erasure_measurement_index_in_list: int):
        for mechanism in self.list_of_mechanisms:
            mechanism.erasure_measurement_index_in_list = erasure_measurement_index_in_list
    def set_single_measurement_sample(self,single_measurement_sample: Union[List,np.array]):
        for mechanism in self.list_of_mechanisms:
            mechanism.single_measurement_sample = single_measurement_sample
    def get_instruction(self, 
                        qubits: Union[List[int], Tuple[int]],
                        mode:str):
        if self.trivial:
            return []
        list_of_args = []
        for mechanism in self.list_of_mechanisms:
            list_of_args += mechanism.get_instruction(qubits=qubits,mode=mode)
        return list_of_args
    
    def get_dynamic_instruction_vectorized(self,
                                           qubits:Union[List[int], Tuple[int]]):
        if self.trivial:
            return []
        list_of_args = []
        for mechanism in self.list_of_mechanisms:
            list_of_args.extend(mechanism.get_dynamic_instruction_vectorized(qubits=qubits))
        return list_of_args
    
    def __repr__(self) -> str:
        s = '''Gate_error_model of the following mechanisms:\n'''
        for i,mech in enumerate(self.list_of_mechanisms):
            s += f"mechanism {i}: \n"
            s += mech.__repr__()
            
        return  s

def product_of_sigma(s1,s2):
    assert s1 in ['X','Y','Z','I'] and s2 in ['X','Y','Z','I']
    return {
        ('I','I'):'I',
        ('I','X'):'X',
        ('I','Y'):'Y',
        ('I','Z'):'Z',

        ('X','I'):'X',
        ('X','X'):'I',
        ('X','Y'):'Z',
        ('X','Z'):'Y',

        ('Y','I'):'Y',
        ('Y','X'):'Z',
        ('Y','Y'):'I',
        ('Y','Z'):'X',

        ('Z','I'):'Z',
        ('Z','X'):'Y',
        ('Z','Y'):'X',
        ('Z','Z'):'I',
    }[(s1,s2)]

def error_mechanism_product(m1:Error_mechanism,m2:Error_mechanism) -> Error_mechanism:
    # Not used anymore, better use a Gate_error_model to represent two independent mechanisms
    num_qubits = m1.num_qubits
    assert num_qubits == m2.num_qubits
    product_list_of_MQE = []
    for event1 in m1.list_of_MQE:
        for event2 in m2.list_of_MQE:
            list_of_SQE = []
            for i in range(num_qubits):
                list_of_SQE.append(SQE(type=product_of_sigma(event1.list_of_SQE[i].type,event2.list_of_SQE[i].type),
                                       heralded=any([event1.list_of_SQE[i].heralded,event2.list_of_SQE[i].heralded])))
            product_list_of_MQE.append(MQE(event1.p * event2.p,list_of_SQE) )
    return Error_mechanism(list_of_MQE=product_list_of_MQE)




def get_1q_depolarization_mechanism(p_p):
    return Error_mechanism(
        [   MQE(1-p_p,[SQE("I",False)]),
            MQE(p_p/3,[SQE("X",False)]),
            MQE(p_p/3,[SQE("Y",False)]),
            MQE(p_p/3,[SQE("Z",False)])
        ]
    )

def get_2q_depolarization_mechanism(p_p):
    return Error_mechanism(
        [   MQE(1-p_p,[SQE("I",False),SQE("I",False)]),

            MQE(p_p/15,[SQE("I",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("I",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("I",False),SQE("Z",False)]),

            MQE(p_p/15,[SQE("X",False),SQE("I",False)]),
            MQE(p_p/15,[SQE("X",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("X",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("X",False),SQE("Z",False)]),

            MQE(p_p/15,[SQE("Y",False),SQE("I",False)]),
            MQE(p_p/15,[SQE("Y",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("Y",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("Y",False),SQE("Z",False)]),

            MQE(p_p/15,[SQE("Z",False),SQE("I",False)]),
            MQE(p_p/15,[SQE("Z",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("Z",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("Z",False),SQE("Z",False)]),
        ]
    )

def get_1q_biased_erasure_with_differential_shift_mechanism(p_e,p_z_shift):
    return Error_mechanism(
    [   
        MQE((1 - p_e )* (1-p_z_shift),[SQE("I",False)]),
        MQE((1 - p_e )* p_z_shift,[SQE("Z",False)]),
        MQE(p_e/2,[SQE("I",True)]),
        MQE(p_e/2,[SQE("Z",True)])
    ]
)

def get_1q_error_model(p_e,p_z_shift, p_p):
    return Gate_error_model([get_1q_biased_erasure_with_differential_shift_mechanism(p_e,p_z_shift),
                             get_1q_depolarization_mechanism(p_p)])


def get_2q_biased_erasure_with_differential_shift_mechanism(p_e,p_z_shift):
    '''
    p_z_shift is the probability of undetectible Z error given not detecting erasure.
    '''
    return Error_mechanism(
    [   
        MQE((1- p_e)**2 * (1-p_z_shift)**2,[SQE("I",False),SQE("I",False)]), # no detection cases
        MQE((1- p_e)**2 * p_z_shift * (1-p_z_shift),[SQE("Z",False),SQE("I",False)]),# differential shift on computational subspace
        MQE((1- p_e)**2 * p_z_shift * (1-p_z_shift),[SQE("I",False),SQE("Z",False)]),
        MQE((1- p_e)**2 * p_z_shift**2,[SQE("Z",False),SQE("Z",False)]),

        MQE(p_e/2 * (1- p_e),[SQE("I",True),SQE("I",False)]), # Single qubit detection cases
        MQE(p_e/2 * (1- p_e),[SQE("Z",True),SQE("I",False)]),

        MQE((1- p_e) * p_e/2,[SQE("I",False),SQE("I",True)]),
        MQE((1- p_e) * p_e/2,[SQE("I",False),SQE("Z",True)]),

        MQE( (p_e/2)**2,[SQE("I",True),SQE("I",True)]), # Two qubit detection cases
        MQE( (p_e/2)**2,[SQE("I",True),SQE("Z",True)]),
        MQE( (p_e/2)**2,[SQE("Z",True),SQE("I",True)]),
        MQE( (p_e/2)**2,[SQE("Z",True),SQE("Z",True)]),
    ]
)



def get_2q_error_model(p_e,p_z_shift, p_p):
    return Gate_error_model([get_2q_biased_erasure_with_differential_shift_mechanism(p_e,p_z_shift),get_2q_depolarization_mechanism(p_p)])


