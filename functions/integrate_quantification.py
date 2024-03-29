def integrate_quantification(value1, value2, value3, value4, weight1, weight2, weight3, weight4):
    # make sure the sum of weighty is 1
    total_weight = weight1 + weight2 + weight3 + weight4
    if total_weight != 1:
        raise ValueError("The sum of weights must be 1.")
    
    # calculate the weighted sum
    integrated_value = value1 * weight1 + value2 * weight2 + value3 * weight3 + value4 * weight4
    
    return integrated_value