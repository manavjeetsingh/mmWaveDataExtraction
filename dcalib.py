import numpy as np

def read_dca1000(file_name, num_adc_samples = 256, num_adc_bits = 16, num_rx = 4):
    """
    Reads the binary file produced by the DCA1000 and Mmwave Studio.
    
    Args:
        file_name (str): Path to the .bin file.
        
    Returns:
        np.ndarray: A numpy array of shape (numRX, numChirps * numADCSamples) containing complex data.
    """
    
    # --- Global Variables (Change based on sensor config) ---
    # num_adc_samples = 256    # number of ADC samples per chirp
    # num_adc_bits = 16        # number of ADC bits per sample
    # num_rx = 4               # number of receivers
    num_lanes = 2            # do not change. number of lanes is always 2
    is_real = False          # set to True if real only data, False if complex data

    # --- Read File ---
    # Read .bin file as int16
    try:
        adc_data = np.fromfile(file_name, dtype=np.int16)
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return None

    # If 12 or 14 bits ADC per sample compensate for sign extension
    if num_adc_bits != 16:
        l_max = 2**(num_adc_bits-1) - 1
        adc_data[adc_data > l_max] -= 2**num_adc_bits

    file_size = adc_data.size

    # --- Process Data ---
    if is_real:
        # Real data reshape
        num_chirps = file_size // (num_adc_samples * num_rx)
        
        # Reshape to match MATLAB's column-major (F) logic
        lvds = adc_data.reshape(num_adc_samples * num_rx, num_chirps, order='F')
        lvds = lvds.T 
    else:
        # Complex data
        # File size is halved because 2 integers make 1 complex number
        num_chirps = file_size // (2 * num_adc_samples * num_rx)
        
        # The DCA1000 usually interleaves data in a specific pattern for 2 lanes:
        # [Real1, Real2, Imag1, Imag2, Real3, Real4, Imag3, Imag4...]
        # We reshape to (-1, 4) to separate these chunks
        adc_data = adc_data.reshape(-1, 4)
        
        # Construct complex data
        # Real parts are columns 0 and 1; Imaginary parts are columns 2 and 3
        data_real = adc_data[:, [0, 1]].flatten()
        data_imag = adc_data[:, [2, 3]].flatten()
        complex_data = data_real + 1j * data_imag
        
        # Reshape to dimensions: (SamplesPerChirp * NumRX) x NumChirps
        # We use order='F' to mimic MATLAB's default column-filling behavior
        lvds = complex_data.reshape(num_adc_samples * num_rx, num_chirps, order='F')
        
        # Transpose to get (NumChirps) x (SamplesPerChirp * NumRX)
        lvds = lvds.T

    # --- Organize Data per RX ---
    # Currently, lvds shape is (num_chirps, num_rx * num_adc_samples)
    # The columns are packed as: [RX1_Samples | RX2_Samples | RX3_Samples | RX4_Samples]
    
    # 1. Reshape to separate RX channels: (NumChirps, NumRX, NumSamples)
    lvds = lvds.reshape(num_chirps, num_rx, num_adc_samples)
    
    # 2. Transpose to bring RX to the front: (NumRX, NumChirps, NumSamples)
    lvds = lvds.transpose(1, 0, 2)
    
    # 3. Flatten the last two dimensions to get continuous samples per RX
    # Final Shape: (NumRX, NumChirps * NumSamples)
    ret_val = lvds.reshape(num_rx, -1)

    return ret_val

# Example Usage:
# radar_data = read_dca1000('adc_data.bin')
# print(radar_data.shape)