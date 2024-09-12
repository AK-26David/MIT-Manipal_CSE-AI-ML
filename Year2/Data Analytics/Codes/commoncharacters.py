string1=str(input("Enter the first string:"))
string2=str(input("Enter the second string:"))
if(len(string1)!=len(string2)):
    print("Cannot find common characters.")
else:
    n=len(string1)
    for i in range(1,n):
        if(string1[i]==string2[i]):
            print(string1[i])