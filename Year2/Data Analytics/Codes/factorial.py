num = int(input("Enter a number for calculation:"))
factorial =1
if num<0:
    print("Factorial cannot be found.")
elif num==0:
    print("The factorial of 0 is 1.")
else:
    for i in range(1,num+1):
        factorial=factorial*i
        
    print(f"The factorial of {num} is {factorial}")