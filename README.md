# surface_erasure_decoding
 
The rotated surface code part is derived from [Stim](https://github.com/quantumlib/Stim)'s c++ code

This package help me generate decoding problem istances that I can send to distributed computing nodes. The method I'm using to decode erasure is inefficient because it is 1) python based 2) doesn't use importance sampling 3) uses existing MWPM library.
