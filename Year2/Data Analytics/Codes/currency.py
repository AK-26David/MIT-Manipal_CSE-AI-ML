def main():
    currency_dictionary = {}
    n = int(input("Enter the number of inputs that are required:"))
    
    for i in range(n):
        dict_key = input("Enter key:")
        dict_value = input("Enter value:")  # Corrected this line
        currency_dictionary[dict_key] = dict_value
    
    print(currency_dictionary)
    
    # To traverse through the dictionary
    for key, value in currency_dictionary.items():  # Corrected this loop
        print(f"{key} has the currency {value}")

if __name__ == "__main__":
    main()
