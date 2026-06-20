#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// System Parameters
const int N = 20000000;          // Number of samples (large enough to stress the CPU during convolution)
const int FILTER_TAPS = 1000;   // Low Pass Filter (LPF) size
const float Fc = 10000.0f;     // Carrier frequency
const float Fs = 100000.0f;    // Sampling frequency
const float LNA_GAIN = 10.0f;  // Low Noise Amplifier (LNA) Gain

int main() {
    // 1. Memory Allocation
    std::vector<float> Y_ant(N);
    std::vector<float> Y_lna(N);
    std::vector<float> Y_demod(N);
    std::vector<float> Y_filtered(N);
    
    // Simplified LPF coefficients (Moving average for this example)
    std::vector<float> LPF_coeffs(FILTER_TAPS, 1.0f / FILTER_TAPS); 

    // Noise Generators (AWGN)
    std::mt19937 gen(42); // Fixed seed for reproducible benchmarking
    std::normal_distribution<float> channel_noise(0.0f, 0.1f);
    std::normal_distribution<float> lna_noise(0.0f, 0.05f);

    // --- START TIME MEASUREMENT ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // STEP 1: Generate Y_ant(t) - DSB Signal Transmission + Channel Noise
    for (int i = 0; i < N; ++i) {
        float t = (float)i / Fs;
        float message = std::cos(2.0f * M_PI * 1000.0f * t); // Information signal
        float carrier = std::cos(2.0f * M_PI * Fc * t);      // Carrier wave
        
        // DSB modulation (creates the two lobes in the spectrum)
        float dsb_signal = message * carrier;
        Y_ant[i] = dsb_signal + channel_noise(gen); 
    }

    // STEP 2: Pass through Low Noise Amplifier (LNA)
    for (int i = 0; i < N; ++i) {
        Y_lna[i] = (Y_ant[i] * LNA_GAIN) + lna_noise(gen);
    }

    // Demodulation (Bring the useful lobe to baseband)
    for (int i = 0; i < N; ++i) {
        float t = (float)i / Fs;
        float local_oscillator = std::cos(2.0f * M_PI * Fc * t);
        Y_demod[i] = Y_lna[i] * local_oscillator;
    }

    // Apply LPF to cut off the lobe at 2*Fc (Convolution)
    // CAUTION: This is the code bottleneck with O(N * FILTER_TAPS) complexity
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < FILTER_TAPS; ++j) {
            if (i - j >= 0) {
                sum += Y_demod[i - j] * LPF_coeffs[j];
            }
        }
        Y_filtered[i] = sum;
    }

    // Calculate Signal Power and apply SNR formula
    float signal_power = 0.0f;
    float noise_power = 0.0f;
    
    for (int i = 0; i < N; ++i) {
        signal_power += Y_filtered[i] * Y_filtered[i];
        noise_power += 0.01f; // Constant noise estimation to simplify the example
    }
    
    signal_power /= N;
    noise_power /= N;
    float snr = 10.0f * std::log10(signal_power / noise_power);

    // --- END TIME MEASUREMENT ---
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "--- Serial Execution (Baseline) ---" << std::endl;
    std::cout << "Sample Size (N): " << N << std::endl;
    std::cout << "SNR: " << snr << " dB" << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}