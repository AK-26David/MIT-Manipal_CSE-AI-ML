import math
n=int(input("Enter a number:"))
m=int(math.sqrt(n))
for i in range(2,m):
    if m%i==0:
        prime=False
        break
    
if prime:
    print(f"Square root is prime!")
else:
    print(f"Square root is not prime!")
    