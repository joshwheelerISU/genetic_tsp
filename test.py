import random

def get_random_paths( population):
      # 3 variables, can be adjusted to accommodate returning a larger
      # amount of samples from the population
    a, b, c = 0, 0, 0
       # select random samples from the population pool.
    while a == b  or b == c or a == c:
        a = random.randint(0, len(population) - 1)
        b = random.randint(0, len(population) - 1)
        c = random.randint(0, len(population) - 1)
            # we now have 3 unique indexes from the population pool, so return them
    return population[a], population[b], population[c]


print(get_random_paths([0,1,2,3,4,5,6,7,8,9]))