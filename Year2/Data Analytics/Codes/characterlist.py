def main():
    n = int(input("Enter an index number:"))
    List = [int(x) for x in input("Enter the characters in the list:").split()]
    print("First 3 elements:", List[:3])  # Slicing to get the first 3 elements
    middle_index = len(List) // 2
    print("Middle elements:", List[middle_index:])  # Slicing to get the middle and tail elements
    print("Elements from index", n, "to the end:", List[n:])

if __name__ == "__main__":
    main()
