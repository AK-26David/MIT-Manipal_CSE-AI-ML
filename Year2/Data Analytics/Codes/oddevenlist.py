def odd(listodd):
    print("The odd list is:")
    for i in range(len(listodd)):
        if listodd[i] % 2 != 0:
            print(listodd[i])

def even(listeven):
    print("The even list is:")
    for i in range(len(listeven)):
        if listeven[i] % 2 == 0:
            print(listeven[i])

def main():
    input_string = input("Enter integers in the given list, separated by spaces: ")
    listmain = [int(x) for x in input_string.split()]
    odd(listmain)
    even(listmain)

if __name__ == "__main__":
    main()