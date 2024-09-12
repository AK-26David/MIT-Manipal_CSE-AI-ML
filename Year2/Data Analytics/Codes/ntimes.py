def print_String(str1,num):
    for i in range(1,num):
        print(str1)
        print("\n")
def main():
    str1=str(input("Enter a string:"))
    num=int(input("Enter a number:"))
    printstr=print_String(str1,num)
    
if __name__=="__main__":
    main()