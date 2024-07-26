from sur_v7_builder import *
from sur_v7_model import *

import time

@dataclass
class Sample_decode_job:
    d: int
    p_e: float
    p_z_shift:float
    p_p: float
    shots: int

    job_id: str
    circuit_id: str

    def get_builder(self):
        # Assemble circuit builder
        after_cz_error_model = get_2q_error_model(p_e=self.p_e,
                                                  p_z_shift=self.p_z_shift, 
                                                  p_p=self.p_p)
        meas_error = 0.001
        builder = easure_circ_builder(rounds = self.d,
                                      distance= self.d,
                                      after_cz_error_model=after_cz_error_model,
                                      measurement_error=meas_error
                                      )
        builder.generate_circuit_and_decoding_info()
        return builder
    
    def sample_and_print_result(self,print_progress = False):
        if print_progress:
            from IPython.display import clear_output

        builder = self.get_builder()        
        sampler = builder.erasure_circuit.compile_sampler()
        meas_samples = sampler.sample(shots=self.shots)
        converter = builder.erasure_circuit.compile_m2d_converter()
        det_samples, actual_obs_chunk = converter.convert(measurements=meas_samples,
                                                                separate_observables=True)
        t1 = time.time()
        # Decode
        new_circ_num_errors = 0
        normal_circ_num_errors = 0
        for i in range(self.shots):
            predicted  = builder.decode_by_generate_new_circ(det_samples[i],'S',meas_samples[i])
            new_circ_num_errors += actual_obs_chunk[i][0] != predicted

            predicted = builder.decode_without_changing_weights(det_samples[i],'S',meas_samples[i])
            normal_circ_num_errors += actual_obs_chunk[i][0] != predicted

            if i%10 == 0 and print_progress:
                clear_output(wait=True)
                print(f'decoding finished {100*i/self.shots}%')
        t2 = time.time()
        if print_progress:
            print(f"{(t2-t1)/self.shots} per shot (d = {self.d})")
        result = {
            'job_id': self.job_id,
            'circuit_id': self.circuit_id,
            'shots':self.shots,
            'new_circ':     int(new_circ_num_errors),
            'normal': int(normal_circ_num_errors)
        }

        return result



# This function can be run over condor
def main():
    counter = sys.argv[1]
    # Load Sample_decode_job instance
    with open(f'{counter}.pkl', 'rb') as f:
        job = pickle.load(f)
    # Get result back from Sample_decode_job instance
    result = job.sample_and_print_result()
    # Print result
    # Just let the printed messaged be loaded into the .out files automatically.
    json_str = json.dumps(result)
    sys.stdout.buffer.write(json_str.encode())

def decode_locally(counter):
    pass


if __name__ == '__main__':
    main()


