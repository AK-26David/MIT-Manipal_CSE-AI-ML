def extract(stringMain,n):
    for i in range(0,n):
        print(stringMain[i])
def main():
    stringMain=str(input("Enter a string:"))
    n=int(input("Enter the number of characters to be extracted:"))
    test=extract(stringMain,n)
    
if __name__=="__main__":
    main()