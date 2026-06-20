#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h> 

#include "lpf_ispc.h" 

const int N = 20000000;        // 20 Million samples
const int FILTER_TAPS = 1000;  // LPF size
const float Fc = 10000.0f;     
const float Fs = 100000.0f;    
const float LNA_GAIN = 10.0f;  

int main() {
    std::vector<float> Y_demod(N);
    std::vector<float> Y_filtered(N);
    std::vector<float> LPF_coeffs(FILTER_TAPS, 1.0f / FILTER_TAPS); 

    std::mt19937 gen(42); 
    std::normal_distribution<float> channel_noise(0.0f, 0.1f);
    std::normal_distribution<float> lna_noise(0.0f, 0.05f);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Loop Fusion (Serial)
    for (int i = 0; i < N; ++i) {
        float t = (float)i / Fs;
        float carrier = std::cos(2.0f * M_PI * Fc * t); 
        float message = std::cos(2.0f * M_PI * 1000.0f * t);
        float y_ant_temp = (message * carrier) + channel_noise(gen); 
        float y_lna_temp = (y_ant_temp * LNA_GAIN) + lna_noise(gen);
        Y_demod[i] = y_lna_temp * carrier; 
    }

    // HYBRID PARALLELISM (OpenMP + ISPC)
    // Divide the 20 million samples into chunks of 4096. 
    // OpenMP distributes the chunks to the CPU cores, ISPC executes them using SIMD.
    int chunk_size = 4096;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += chunk_size) {
        int end_idx = std::min(i + chunk_size, N);
        // Call the ISPC kernel
        ispc::apply_lpf_ispc_chunk(Y_demod.data(), LPF_coeffs.data(), Y_filtered.data(), i, end_idx, FILTER_TAPS);
    }

    // Power Calculation (OpenMP Reduction)
    float signal_power = 0.0f;
    float noise_power = 0.0f;
    
    #pragma omp parallel for reduction(+:signal_power, noise_power)
    for (int i = 0; i < N; ++i) {
        signal_power += Y_filtered[i] * Y_filtered[i];
        noise_power += 0.01f; 
    }
    
    signal_power /= N;
    noise_power /= N;
    float snr = 10.0f * std::log10(signal_power / noise_power);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "--- Hybrid Execution (OpenMP + ISPC) ---" << std::endl;
    std::cout << "SNR: " << snr << " dB" << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}