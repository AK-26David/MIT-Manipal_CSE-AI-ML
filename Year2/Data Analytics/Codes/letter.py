def letter(str1):
    for i in range(0,len(str1)):
        print(str1[i])
        print("\n")
def main():
    str1=str(input("Enter a string:"))
    lett=letter(str1)
    
if __name__=="__main__":
    main()