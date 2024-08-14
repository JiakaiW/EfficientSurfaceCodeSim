# surface_erasure_decoding
 
The rotated surface code part is derived from [Stim](https://github.com/quantumlib/Stim)'s c++ code

This package help me generate decoding problem istances that I can send to distributed computing nodes. The method I'm using to decode erasure is inefficient because it is 1) python based without a specificly optimized erasure handling mechanism 2) doesn't use importance sampling

How I used this package:
 1 use Docker to build a container and store in DockerHub.
 2 generate decoding problem instances and send those instances to distributed computing
 3 gather those decoding results in form of JSON files
 4 data analytics on my local computer
 
TODO: 
1) add importance sampling
