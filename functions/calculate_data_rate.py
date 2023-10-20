import math
from functions.dB_conversion import dB_conversion

def calculate_data_rate(UAVInfo, UAV_position, ground_user_position, block_or_not):
    channel_bandwidth = UAVInfo['bandwidth']
    UAV_power = dB_conversion(UAVInfo['Power'], 'dBm')/1000
    UAV_trans_ante = UAVInfo['TransmittingAntennaGain']
    UAV_rece_ante = UAVInfo['ReceievingAntennaGain']
    UAV_carrier_frequency = UAVInfo['CarrierFrequency']
    
    
    # Constants
    k = 1.38e-23  # Boltzmann's constant
    T = 290  # Noise temperature in Kelvin
    B = channel_bandwidth  # Bandwidth in Hz

    # Parameters
    Pt = UAV_power  # Transmission power of the UAV in Watts
    Gt = UAV_trans_ante  # Gain of the transmitting antenna
    Gr = UAV_rece_ante  # Gain of the receiving antenna
    f = UAV_carrier_frequency  # Frequency of the carrier signal in Hz

    # Calculate the distance between the UAV and the ground user
    d = math.sqrt((UAV_position[0] - ground_user_position[0]) ** 2 +
                  (UAV_position[1] - ground_user_position[1]) ** 2 +
                  (UAV_position[2] - ground_user_position[1]) ** 2)

    # Simplified path loss exponent, may need modification in the future
    n = 4 if block_or_not else 2

    # Wavelength of the carrier signal
    lambda_ = 3e8 / f  # Speed of light is approximately 3 * 10^8 m/s

    # Calculate the received power
    P = Pt * Gt * Gr * (lambda_ / (4 * math.pi * d)) ** n

    # Calculate the noise power
    N = k * T * B

    # Calculate the Signal-to-Noise Ratio (SNR)
    SNR = P / N

    # Calculate the data rate using Shannon Capacity formula
    C = B * math.log2(1 + SNR)

    return C

