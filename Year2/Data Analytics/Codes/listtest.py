def odd(mainList):
    for i in range(0,len(mainList)):
        if mainList[i]%2!=0:
            print(mainList[i])
            
def main():
    mainList = [int(x) for x in input("Enter the elements in the list (space-separated): ").split()]
    test=odd(mainList)
    
if __name__=="__main__":
    main()