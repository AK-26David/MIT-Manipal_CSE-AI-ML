def main():
    build_dictionary={}
    n=int(input("Enter the number of values to be stored:"))
    for i in range(0,n):
        dic_key=input("Enter key:")
        dic_value=input("Enter value:")
        build_dictionary.update({dic_key:dic_value})
        
    print(build_dictionary)
    
if __name__=="__main__":
    main()