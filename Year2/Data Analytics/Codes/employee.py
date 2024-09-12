def main():
    employee_dictionary={}
    n=int(input("Enter the number of inputs required:"))
    for i in range(0,n):
        name=input("Enter name of the employee:")
        employee_id=int(input("Enter the employee id:"))
        salary=int(input("Enter the salary of the employee:"))
        employee_dictionary.update({employee_id:salary})
        sorted_employee_dictionary = dict(sorted(employee_dictionary.items()))

        print(sorted_employee_dictionary)
        
if __name__=="__main__":
    main()