from instruction_vectorizers import *



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
    The parameters used in CORRELATED_ERROR and ELSE_CORRELATED_ERRORs are different from what's given to an Error_mechanism. 
        Error_mechanism converts those probabilities to the type used by CORRELATED_ERROR and ELSE_CORRELATED_ERRORs. See documentation of stim.



    Vectorization:
        different modes have different vectorization rules.
        | Type              | normal mode               |   erasure mode                |   dynamic mode            |   deterministic mode      |
        |-------------------|---------------------------|-------------------------------|---------------------------|---------------------------|
        |1-q nonherald      |Error,q,param              |Error,q,param                  |Error,q,param              |Error,q,param'             |
        |1-q herald         |Error,q,param              |PAULI_CHANNEL_2,[q,a],param    |PAULI_CHANNEL_1,q,param'   |PAULI_CHANNEL_2,q,param'   |
        |2-q nonherald      |Error,[q,p],param          |Error,[q,p],param              |Error,[q,p],param          |Error,[q,p],param'         |
        |2-q herald         |Error,[q,p],param          |PAULI_CHANNEL_2,[q,a],param *2 |PAULI_CHANNEL_1,q,param'*2 |PAULI_CHANNEL_2,[q,p],param'| (there's no heralded 2-qubit errors, decompose them into 1-q heralds)
    """
    normal_vectorizer: NormalInstructionVectorizer
    erasure_vectorizer: Optional[ErasureInstructionVectorizer] = None
    dynamic_vectorizer: Optional[DynamicInstructionVectorizer] = None
    deterministic_vectorizer: Optional[DeterministicInstructionVectorizer] = None

    next_ancilla_qubit_index_in_list: Optional[int] = None
    erasure_measurement_index_in_list: Optional[int] = None
    single_measurement_sample:  Optional[Union[List,np.array]] = None
    name:Optional[str] = None
    def __post_init__(self):
        self.num_qubits = self.normal_vectorizer.num_qubits
        if self.dynamic_vectorizer is None or sum(self.dynamic_vectorizer.p_herald) == 0:
            self.is_erasure = False
        else:
            self.is_erasure = True
        if self.erasure_vectorizer is None or self.dynamic_vectorizer is None:
            self.erasure_vectorizer = self.normal_vectorizer
            self.dynamic_vectorizer = self.normal_vectorizer

    def get_instruction(self, 
                        qubits: Union[List[int], Tuple[int]],
                        mode:str):
        '''
        return list of args that can be used in  stim.circuit.append()

        This function is a newer implementation of generating instructions with posterior probabilities. 
        The vectorization is over a batch of operations. For example, rather than doing one CNOT at a time, this vectorized method 
            generates one batch of CNOT instructions at a time.
        Accept qubits in the style of stim instructions, len(qubits) == integer multiple of self.num_qubits

        # TODO: do I have issue with this:
        # fill_values = fill_values.reshape(num_parallel, self.num_herald_locations).T # is this working? 
        fill_values = fill_values.reshape( self.num_herald_locations,num_parallel) # What used to work before I change to data_qubits_array = np.array(qubits).reshape(-1,self.num_qubits).T
        '''
        

        if mode == 'normal' or self.is_erasure == False: 
            instructions = self.normal_vectorizer.get_instruction(qubits=qubits)
        elif mode == 'erasure':
            instructions =  self.erasure_vectorizer.get_instruction(qubits,self.next_ancilla_qubit_index_in_list)
        elif mode == 'dynamic':
            instructions =  self.dynamic_vectorizer.get_instruction(qubits,self.erasure_measurement_index_in_list,self.single_measurement_sample)
        elif mode == 'deterministic':
            instructions =  self.deterministic_vectorizer.get_instruction(qubits)
        else:
            raise Exception("unsupported mode")
        
        # with open('log.txt', 'a') as file:
        #     # Write the instructions to the file, starting on a new line
        #     file.write('\n name:'+self.name+'  mode:'+mode+'\n')
        #     file.write(str(instructions))

        return instructions


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

    
    def set_next_ancilla_qubit_index_in_list(self,next_ancilla_qubit_index_in_list: int):
        for mechanism in self.list_of_mechanisms:
            mechanism.next_ancilla_qubit_index_in_list = next_ancilla_qubit_index_in_list
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
    
    
    def __repr__(self) -> str:
        s = '''Gate_error_model of the following mechanisms:\n'''
        for i,mech in enumerate(self.list_of_mechanisms):
            s += f"mechanism {i}: \n"
            s += mech.__repr__()
            
        return  s




def get_1q_depolarization_mechanism(p_p):
    list_of_MQE=  [   
                MQE(1-p_p,[SQE("I",False)]),
                MQE(p_p/3,[SQE("X",False)]),
                MQE(p_p/3,[SQE("Y",False)]),
                MQE(p_p/3,[SQE("Z",False)])
            ]
    normal_vectorizer = NormalInstructionVectorizer(
            list_of_MQE = list_of_MQE,
            instruction_name ='DEPOLARIZE1',
            instruction_arg = p_p)

    return Error_mechanism(
        normal_vectorizer = normal_vectorizer,
        name = '1q depo'
        )

def get_1q_differential_shift_mechanism(p_z_shift):
    list_of_MQE= [   
            MQE(1-p_z_shift,[SQE("I",False)]),
            MQE(p_z_shift,[SQE("Z",False)])
        ]
    normal_vectorizer = NormalInstructionVectorizer(
            list_of_MQE = list_of_MQE,
            instruction_name ='Z_ERROR',
            instruction_arg = p_z_shift)

    return Error_mechanism(
        normal_vectorizer= normal_vectorizer,
        name = '1q z shift'
        )

def get_1q_biased_erasure_mechanism(p_e):
    list_of_MQE=  [   
            MQE(1 - p_e,[SQE("I",False)]),
            MQE(p_e/2,[SQE("I",True)]),
            MQE(p_e/2,[SQE("Z",True)])
        ]
    normal_vectorizer = NormalInstructionVectorizer(
            list_of_MQE = list_of_MQE,
            instruction_name ='Z_ERROR',
            instruction_arg = p_e/2)
    erasure_vectorizer = ErasureInstructionVectorizer(
            list_of_MQE = list_of_MQE,
            instruction_name = "PAULI_CHANNEL_2",
            instruction_arg = [
                  # ix iy iz
                    p_e / 2, 0, 0,
                  # xi xx xy xz
                    0, 0, 0, 0,
                  # yi yx yy yz
                    0, 0, 0, 0,
                  # zi zx zy zz
                    0, p_e / 2, 0, 0
                ])
    dynamic_vectorizer = DynamicInstructionVectorizer(
            list_of_MQE = list_of_MQE
            )
    return Error_mechanism(
        normal_vectorizer = normal_vectorizer,
        erasure_vectorizer=erasure_vectorizer,
        dynamic_vectorizer = dynamic_vectorizer,
        name = '1q erasure'
        )

def get_1q_error_model(p_e,p_z_shift, p_p):
    mechanism_list = [get_1q_depolarization_mechanism(p_p)]
    if p_z_shift>0:
        mechanism_list.append(get_1q_differential_shift_mechanism(p_z_shift))
    if p_e>0:
        mechanism_list.append(get_1q_biased_erasure_mechanism(p_e))
    return Gate_error_model(mechanism_list)




def get_2q_depolarization_mechanism(p_p):
    list_of_MQE= [
           MQE(1-p_p,[SQE("I",False),SQE("I",False)]),

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
    normal_vectorizer = NormalInstructionVectorizer(
            list_of_MQE = list_of_MQE,
            instruction_name ='DEPOLARIZE2',
            instruction_arg = p_p)
    return Error_mechanism(
        normal_vectorizer=normal_vectorizer,
        name = '2q depo'
        )

def get_2q_differential_shift_mechanism(p_z_shift):
    list_of_MQE= [
            MQE((1-p_z_shift)**2,[SQE("I",False),SQE("I",False)]),
            MQE(p_z_shift * (1-p_z_shift),[SQE("Z",False),SQE("I",False)]),
            MQE(p_z_shift * (1-p_z_shift),[SQE("I",False),SQE("Z",False)]),
            MQE(p_z_shift**2,[SQE("Z",False),SQE("Z",False)]),
        ]
    normal_vectorizer = NormalInstructionVectorizer(
        list_of_MQE = list_of_MQE,
        instruction_name ='Z_ERROR',
        instruction_arg = p_z_shift)
    return Error_mechanism(
        normal_vectorizer= normal_vectorizer,
        name = '2q z shift'
        )

def get_2q_biased_erasure_mechanism(p_e):
    list_of_MQE=  [   
                MQE((1- p_e)**2,[SQE("I",False),SQE("I",False)]), # no detection cases

                MQE(p_e/2 * (1- p_e),[SQE("I",True),SQE("I",False)]), # Single qubit detection cases
                MQE(p_e/2 * (1- p_e),[SQE("Z",True),SQE("I",False)]),
                MQE((1- p_e) * p_e/2,[SQE("I",False),SQE("I",True)]),
                MQE((1- p_e) * p_e/2,[SQE("I",False),SQE("Z",True)]),

                MQE( (p_e/2)**2,[SQE("I",True),SQE("I",True)]), # Two qubit detection cases
                MQE( (p_e/2)**2,[SQE("I",True),SQE("Z",True)]),
                MQE( (p_e/2)**2,[SQE("Z",True),SQE("I",True)]),
                MQE( (p_e/2)**2,[SQE("Z",True),SQE("Z",True)]),
            ]
    normal_vectorizer = NormalInstructionVectorizer(
        list_of_MQE = list_of_MQE,
        instruction_name ='Z_ERROR',
        instruction_arg = p_e/2)
    erasure_vectorizer = ErasureInstructionVectorizer(
        list_of_MQE = list_of_MQE,
        instruction_name = "PAULI_CHANNEL_2",
        instruction_arg = [
                # ix iy iz
                p_e / 2, 0, 0,
                # xi xx xy xz
                0, 0, 0, 0,
                # yi yx yy yz
                0, 0, 0, 0,
                # zi zx zy zz
                0, p_e / 2, 0, 0
            ])
    dynamic_vectorizer = DynamicInstructionVectorizer(
            list_of_MQE = list_of_MQE
            )
    return Error_mechanism(
        normal_vectorizer = normal_vectorizer,
        erasure_vectorizer=erasure_vectorizer,
        dynamic_vectorizer = dynamic_vectorizer,
        name = '2q erasure'
        )


def get_2q_error_model(p_e,p_z_shift, p_p):
    mechanism_list = [get_2q_depolarization_mechanism(p_p)]
    if p_z_shift>0:
        mechanism_list.append(get_2q_differential_shift_mechanism(p_z_shift))
    if p_e>0:
        mechanism_list.append(get_2q_biased_erasure_mechanism(p_e))
    return Gate_error_model(mechanism_list)

