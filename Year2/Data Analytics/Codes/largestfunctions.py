def largest(num1,num2,num3):
    if num1>num2 and num1>num3:
        print(f"{num1} is the largest number!!!")
    elif num2>num1 and num2>num3:
        print(f"{num2} is the largest number!!!")
    else:
        print(f"{num3} is the largest number!!!")
        
def main():
    num1=int(input("Enter the first number:"))
    num2=int(input("Enter the second number:"))
    num3=int(input("Enter the third number:"))
    large=largest(num1,num2,num3)
    
if __name__=="__main__":
    main()
    
        