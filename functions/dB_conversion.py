def dB_conversion(old_data, old_unit):
    """
    Convert dB data into basic unit. For example, 0 dBm -> 1mW.
    """
    if old_unit == 'dBm':
        # If the old unit is 'dBm', the output is in the unit of 'mW'
        return 10 ** (old_data / 10)
    else:
        print('Unknown unit')
        return None
