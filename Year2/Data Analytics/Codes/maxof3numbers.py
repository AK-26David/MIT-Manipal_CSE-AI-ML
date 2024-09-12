def max_of_three(num1, num2, num3):
    if num1 > num2 and num1 > num3:
        return num1
    elif num2 > num1 and num2 > num3:
        return num2
    else:
        return num3

def main():
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    num3 = int(input("Enter the third number: "))

    maximum = max_of_three(num1, num2, num3)
    print(f"The greatest number of the three is {maximum}.")

if __name__ == "__main__":
    main()
