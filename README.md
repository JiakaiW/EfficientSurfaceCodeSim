# surface_erasure_decoding
Simulate and decoder surface code with erasure (assuming perfect erasure convertion) [Stim](https://github.com/quantumlib/Stim)'s c++ code

Features highly abstracted and customizable noise modelling classes that supports 
1. simulating and decoding perfect erasure conversion
2. injecting fixed number of errors into the circuit
3. vectorization of instruction generation for 1. and 2. by using numpy. The rules for vectorization are stored in these classes: a) posterior instruction generator for decoding erasure and deterministic instruction generator for usage  in importance sampling[link](https://github.com/JiakaiW/surface_erasure_decoding/blob/main/surface_erasure_decoding/instruction_generators.py).

Circuit level noise is abstracted into these layers:
1. A GateErrorModel contains one or more ErrorMechanisms,
2. An ErrorMechanism contains multiple InstructionGenerators that tells the ErrorMechanism how to inject stim instructions in different circuit generation modes
3. InstructionGenerators uses abstraction like MQE (Multi Qubit Error/Event) and SQE (Single Qubit Error/Event) to calculate the posterior probability given leakage detection results.

# Direct Monte-Carlo usage:
1. use Docker to build a container and store in DockerHub.
2. generate decoding problem instances [(job class)](surface_erasure_decoding/job.py)) and send those instances to distributed computing
3. gather those decoding results in form of JSON files
4. data analytics on my local computer

# Importance sampling usage (injecting fixed number of errors in random locations):
1. Use "deterministic" mode when generating the circuit, and decode as usual. (call decode_by_generating_new_circuit if erasure conversion is involved.)
2. Currently we don't have a solution to efficiently estimate the logical error rate (WIP).
