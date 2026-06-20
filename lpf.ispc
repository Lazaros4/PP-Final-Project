// lpf.ispc

export void apply_lpf_ispc_chunk(uniform float Y_demod[], 
                                 uniform float LPF_coeffs[], 
                                 uniform float Y_filtered[], 
                                 uniform int start_idx, 
                                 uniform int end_idx, 
                                 uniform int FILTER_TAPS) 
{

    foreach (i = start_idx ... end_idx) {
        float sum = 0.0f;
        
        for (uniform int j = 0; j < FILTER_TAPS; ++j) {
            if (i >= j) {
                sum += Y_demod[i - j] * LPF_coeffs[j];
            }
        }
        Y_filtered[i] = sum;
    }
}