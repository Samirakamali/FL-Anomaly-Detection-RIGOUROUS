def weighted_average_time(response_times):

    response_times.sort(reverse=True)
    weights = []
    for time in response_times:
      if time != 0:
        weights.append(1 / time) 
      else:
        weights.append(1)  # Assign a weight of 1 for zero response times
    weights = weights[::-1]
    total_weight = sum(weights)
    if total_weight == 0:
      return None
    weighted_times = [time * weight for time, weight in zip(response_times, weights)]  
    weighted_average_time = sum(weighted_times) / total_weight  

    return weighted_average_time