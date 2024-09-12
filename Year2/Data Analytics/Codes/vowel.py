def vowel(String):
    vowels = "aeiouAEIOU"
    for char in String:
        if char in vowels:
            print(char)
                            
def main():
    String=str(input("Enter a string:"))
    Vowel=vowel(String)
    
if __name__=="__main__":
    main()