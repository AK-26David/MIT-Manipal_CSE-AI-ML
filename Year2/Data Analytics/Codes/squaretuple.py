def create_tuple_list(start, end):
    if start > end:
        start, end = end, start  # Swap start and end if start is greater

    tuple_list = [(x, x ** 2) for x in range(start, end + 1)]

    return tuple_list

# Input range from the user
start = int(input("Enter the start of the range: "))
end = int(input("Enter the end of the range: "))

result = create_tuple_list(start, end)

# Display the list of tuples
print("List of tuples within the range:")
for tup in result:
    print(tup)

    