def replace(String,Character):
    for i in range(0,len(String)):
        if String[i]!=Character:
            print(String[i])
            
def main():
    String=str(input("Enter a string:"))
    Character=input("Enter a character:")
    test=replace(String,Character)
    
if __name__=="__main__":
    main()