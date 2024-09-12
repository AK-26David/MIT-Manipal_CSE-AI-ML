ll = int(input("Enter the lower limit: "))
ul = int(input("Enter the upper limit: "))

while ll < ul:
    if ll % 2 == 0:
        print(ll)
    ll += 2  # Increment ll by 2 to move to the next even number
