n = int(input("Enter a number for calculation:"))
sum = 0  # Initialize the sum variable

if n <= 0:
    print("Please enter a positive integer.")
else:
    for i in range(1, n):
        sum += 1 / i

    print(f"The sum is {sum}")
