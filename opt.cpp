#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// System Parameters - Scaled up for proper benchmarking
const int N = 20000000;        // 20 Million samples
const int FILTER_TAPS = 1000;  // Much sharper/heavier LPF
const float Fc = 10000.0f;     // Carrier frequency
const float Fs = 100000.0f;    // Sampling frequency
const float LNA_GAIN = 10.0f;  // LNA Gain

int main() {
    // Memory Allocation 
    std::vector<float> Y_demod(N);
    std::vector<float> Y_filtered(N);
    
    std::vector<float> LPF_coeffs(FILTER_TAPS, 1.0f / FILTER_TAPS); 

    std::mt19937 gen(42); 
    std::normal_distribution<float> channel_noise(0.0f, 0.1f);
    std::normal_distribution<float> lna_noise(0.0f, 0.05f);

    std::cout << "Starting execution with N = " << N << "..." << std::endl;

    // --- START TIME MEASUREMENT ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // LOOP FUSION (Steps 1, 2, and 3 are now οne loop)
    for (int i = 0; i < N; ++i) {
        float t = (float)i / Fs;
        
        // Step 1: Generate Transmission
        float message = std::cos(2.0f * M_PI * 1000.0f * t);
        float carrier = std::cos(2.0f * M_PI * Fc * t); 
        float y_ant_temp = (message * carrier) + channel_noise(gen); 
        
        // Step 2: Pass through LNA
        float y_lna_temp = (y_ant_temp * LNA_GAIN) + lna_noise(gen);
        
        // Step 3: Demodulation 
        // (Algorithmic optimization: reused 'carrier' instead of recalculating LO)
        Y_demod[i] = y_lna_temp * carrier; 
    }

    // Apply LPF (Convolution) - This will be extremely slow in serial mode!
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < FILTER_TAPS; ++j) {
            if (i - j >= 0) {
                sum += Y_demod[i - j] * LPF_coeffs[j];
            }
        }
        Y_filtered[i] = sum;
    }

    // Calculate Power and SNR
    float signal_power = 0.0f;
    float noise_power = 0.0f;
    
    for (int i = 0; i < N; ++i) {
        signal_power += Y_filtered[i] * Y_filtered[i];
        noise_power += 0.01f; 
    }
    
    signal_power /= N;
    noise_power /= N;
    float snr = 10.0f * std::log10(signal_power / noise_power);

    // --- END TIME MEASUREMENT ---
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "--- Optimized Memory Baseline ---" << std::endl;
    std::cout << "SNR: " << snr << " dB" << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}