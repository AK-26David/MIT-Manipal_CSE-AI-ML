#Write a program to store the latitude and longitude of your house as a tuple and#
#display it.#

latitude=int(input("Enter the latitutde:"))
longitude=int(input("Enter the longitude"))
tupleTest=tuple((latitude,longitude))
print("Coordinates:",{tupleTest})