def data_unit_conversion(data, original_unit):
    """
    Convert some unreadable data into better-looking data with a reasonable unit.
    """
    
    # Conversion factors for different units
    conversion_factors = {
        'k': 1e3,
        'M': 1e6,
        'G': 1e9
    }

    # Determine the appropriate unit for the data
    unit = ''
    if data >= conversion_factors['G']:
        data /= conversion_factors['G']
        unit = 'G'
    elif data >= conversion_factors['M']:
        data /= conversion_factors['M']
        unit = 'M'
    elif data >= conversion_factors['k']:
        data /= conversion_factors['k']
        unit = 'k'

    # Constructing the converted unit
    converted_unit = unit + original_unit

    # Return the converted data and unit
    return data, converted_unit