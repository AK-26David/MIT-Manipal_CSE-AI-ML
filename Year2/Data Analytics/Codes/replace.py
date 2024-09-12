def replace(mainString,subString,replaceString):
    parts=mainString.split(subString)
    replaceString=replaceString.join(parts)
    return replaceString
def main():
    mainString=str(input("Enter a string:"))
    subString=str(input("Enter the string to be replaced:"))
    replaceString=str(input("Enter the replacement string:"))
    result=replace(mainString,subString,replaceString)
    print(result)
    
if __name__=="__main__":
    main()