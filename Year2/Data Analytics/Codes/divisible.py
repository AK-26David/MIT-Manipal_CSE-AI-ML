def divfive(listdivfive):
    print("The list after running the divisibility test by 5 is:")
    for i in range(len(listdivfive)):
        if listdivfive[i] % 5 == 0:
            print(listdivfive[i])

def divsix(listdivsix):
    print("The list after running the divisibility test by 6 is:")
    for i in range(len(listdivsix)):
        if listdivsix[i] % 6 == 0:
            print(listdivsix[i])

def main():
    input_str = input("Enter the numeric elements in the list separated by spaces: ")
    listmain = [int(x) for x in input_str.split()]
    divfive(listmain)
    divsix(listmain)

if __name__ == "__main__":
    main()
