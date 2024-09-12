x = int(input("Enter the first number:"))
y = int(input("Enter the second number:"))
sum = 0

while x <= y:
    sum += x
    x = x + 2

print(f"The sum is {sum}")  # Print the sum after the loop
