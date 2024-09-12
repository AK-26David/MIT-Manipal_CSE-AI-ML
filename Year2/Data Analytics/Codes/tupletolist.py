# Sample list of tuples
list_of_tuples = [(1, 'a'), (2, 'b'), (3, 'c')]

# Using a for loop to unzip
unzipped_list1, unzipped_list2 = [], []

for tup in list_of_tuples:
    unzipped_list1.append(tup[0])
    unzipped_list2.append(tup[1])

# Using list comprehensions to unzip
unzipped_list1 = [tup[0] for tup in list_of_tuples]
unzipped_list2 = [tup[1] for tup in list_of_tuples]

print("Unzipped list 1:", unzipped_list1)
print("Unzipped list 2:", unzipped_list2)
