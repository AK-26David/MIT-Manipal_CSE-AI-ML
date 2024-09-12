def removeVowels(stringMain):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    result = ''

    for char in stringMain:
        if char not in vowels:
            result += char

    return result

def main():
    stringMain = input("Enter a string:")
    result = removeVowels(stringMain)
    print("String with vowels removed:", result)

if __name__ == "__main__":
    main()
