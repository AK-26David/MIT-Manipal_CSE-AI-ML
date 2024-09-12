def harmonic_series(n):
    sum = 0
    for i in range(1,n):
        sum+=1/i

    return sum
        
def main():
    n= int(input("Enter the number for calculation:"))
    hs= harmonic_series(n)
    print(f"The sum is {hs}")
    
if __name__=="__main__":
    main()