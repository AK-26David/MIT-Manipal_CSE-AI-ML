def insert(mainString,subString):
    midString= len(mainString)//2
    modifiedString=mainString[:midString]+subString+mainString[midString:]
    return modifiedString
    
    
def main():
    mainString=str(input("Enter a string:"))
    subString=str(input("Enter a substring to be inserted:"))
    test=insert(mainString,subString)
    print(test)
    
if __name__=="__main__":
    main()