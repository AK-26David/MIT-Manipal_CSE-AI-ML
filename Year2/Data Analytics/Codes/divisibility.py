def divisibility(listMain):
    index = 0  # Initialize the index
    for item in listMain:
        if index % 5 == 0 and index % 6 == 0:
            print(index)
        index += 1  # Increment the index for the next iteration

def main():
    listMain = [int(x) for x in input("Enter the elements in the list:").split()]
    divisibility(listMain)

if __name__ == "__main__":
    main()
