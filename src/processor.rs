use realfft::num_traits::{Float, Signed, Zero};
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal, FftNum, num_complex::*, num_traits::*};
use std::sync::Arc;
use ::{STFT, WindowType};
use std::ops::{Range, AddAssign};
use FromF64;


/// An input/output style audio buffer processor using FFT and then IFFT.
pub struct STFTProcessor<T: Float + Signed + Zero + FftNum + FromF64 + AddAssign> {
    window_size: usize,
    step_size: usize,
    stft: STFT<T>,
    ifft: Arc<dyn ComplexToReal<T>>,
    zero_dummy: Vec<Complex<T>>,
    processor_output: Vec<Complex<T>>,
    transfer_output: Vec<T>,
    overflow_remaining: Range<usize>,
}

impl<T: Float + Signed + Zero + FftNum + FromF64 + AddAssign> STFTProcessor<T> {
    pub fn new(window_type: WindowType, window_size: usize, step_size: usize) -> Self {
        let mut inverse_planner = RealFftPlanner::<T>::new();
        let mut ifft = inverse_planner.plan_fft_inverse(window_size);

        let zero_dummy= ifft.make_input_vec();
        let processor_output = ifft.make_input_vec();
        let transfer_output = ifft.make_output_vec();
        Self {
            window_size,
            step_size,
            stft: STFT::new(window_type, window_size, step_size),
            ifft,
            zero_dummy,
            processor_output,
            transfer_output,
            overflow_remaining: 0..0,
        }
    }

    pub fn process(&mut self, chunk: &mut [T], proc_func: &mut dyn FnMut(&mut[Complex<T>], &mut[Complex<T>], usize) -> ()) {
        self.stft.append_samples(chunk);

        // clear chunk
        chunk.fill(T::zero());

        // dump remaining overflow from transfer_output
        let mut ip = 0usize;
        for ov_i in self.overflow_remaining.clone() {
            chunk[ip] += self.transfer_output[ov_i];
            ip += 1;
        }

        let mut chunk_cursor = 0usize;
        while self.stft.contains_enough_to_compute() {
            self.stft.compute_into_complex_output();

            // Reset processor output.
            self.processor_output.copy_from_slice(self.zero_dummy.as_slice());

            // Process complex output.
            proc_func(self.stft.complex_output.as_mut_slice(), self.processor_output.as_mut_slice(), self.window_size);

            // Apply IFFT on processor output, write into transfer output.
            self.ifft.process(self.processor_output.as_mut_slice(), self.transfer_output.as_mut_slice());

            // output window
            if (chunk_cursor+self.window_size) >= chunk.len() {
                // make overflow.
                // !! there should not be more than one window in the overflow,
                // !! but this is an error that should be caught in the future.

                self.overflow_remaining = chunk_cursor..chunk_cursor+self.window_size;

            }else{
                // output window to chunk
                for i in chunk_cursor..chunk_cursor+self.window_size {
                    chunk[i] += self.transfer_output[i-chunk_cursor];
                }
                chunk_cursor += self.step_size;

            }

        }
    }

}