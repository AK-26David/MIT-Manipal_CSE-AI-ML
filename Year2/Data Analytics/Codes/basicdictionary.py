def main():
    dictionary_name={}
    n=int(input("Enter the number of inputs requried"))
    for i in range(0,n):
        dict_key=input("Enter key:")
        dict_value=input("Enter value:")
        dictionary_name.update({dict_key:dict_value})
    print(dictionary_name)
    
if __name__=="__main__":
    main()