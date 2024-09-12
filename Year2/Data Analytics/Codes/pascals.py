def pascals_trinagle(n):
    for i in range(n+1):
        for j in range(n-i):
            print(' ', end='')
            C = 1
            for j in range(1, i+1):
                print(C, ' ', sep='', end='')
                C = C * (i - j) // j
                print()
           


def main():
    n= int(input("Enter the number of iterations you need in the triangle:"))
    pt= pascals_trinagle(n)
    
if __name__ == "__main__":
    main()

