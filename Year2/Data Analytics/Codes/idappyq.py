def calc(n,dict_key,dict_value):
    if dict_value<40:
        print("E")
def main():
    student={}
    n=int(input("Enter the number of inputs required:"))
    for i in range(1,n):
        dict_key=input("Enter roll number:")
        dict_value=input("Enter marks:")