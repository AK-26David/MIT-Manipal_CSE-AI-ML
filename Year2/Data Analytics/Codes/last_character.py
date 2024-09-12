def last_character(string1):
    print(string1[len(string1)-1])
    
def main():
    string1=str(input("Enter a string:"))
    lc=last_character(string1)
    
if __name__=="__main__":
    main()
    