def main():
    dictionary_country = {}
    n = int(input("Enter the total number of inputs required:"))

    for i in range(n):
        dict_key = input("Enter key:")
        dict_value = input("Enter value:")
        dictionary_country.update({dict_key: dict_value})

    sorted_dictionary = dict(sorted(dictionary_country.items()))
    print("Sorted dictionary by keys:", sorted_dictionary)

if __name__ == "__main__":
    main()
