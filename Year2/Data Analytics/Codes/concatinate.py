def concatinate(str1,str2):
    result="{}{}".format(str1,str2)
    return result
def main():
    str1=str(input("Enter the first string:"))
    str2=str(input("Enter the second string:"))
    test=concatinate(str1,str2)
    print(test)
    
if __name__=="__main__":
    main()