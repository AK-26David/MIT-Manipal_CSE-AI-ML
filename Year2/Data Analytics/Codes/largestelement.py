def largest(mainList):
    max_element = max(mainList)
    print("The largest element in the list is:", max_element)

def main():
    mainList = [int(x) for x in input("Enter elements in the list:").split()]
    largest(mainList)

if __name__ == "__main__":
    main()
