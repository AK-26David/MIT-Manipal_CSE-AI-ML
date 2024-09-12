import math

n = int(input("Enter the limit for calculation:"))
total_sum = 0  # Initialize the total sum to 0

while n > 0:
    numerator = pow(n, n)
    denominator = n
    term = numerator / denominator
    total_sum += term  # Accumulate the sum
    n -= 1  # Decrement n in each iteration

print(f"The sum is {total_sum}")
