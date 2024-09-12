def harmonic_progression(num):
    sum=0
    for i in range(1,num):
        term=1/i
        sum+=term
    print(f"The sum of the sries is {sum}")

def main():
    num=int(input("Enter the limit for calculation:"))
    hp=harmonic_progression(num)
    
if __name__=="__main__":
    main()