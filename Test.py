import random

# For 3 random numbers
random_numbers = random.sample(range(0, 15), 11)
print(sorted(random_numbers))

# For 4 random numbers
random_numbers = random.sample(range(1, 100), 4)
print(random_numbers)

# For 5 random numbers
random_numbers = random.sample(range(1, 100), 5)
print(random_numbers)
