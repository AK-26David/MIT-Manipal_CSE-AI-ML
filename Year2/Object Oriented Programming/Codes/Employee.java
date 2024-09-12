import java.util.Scanner;
class employee
{
    int empNum;
    double empSalary;
    employee(int empNum,double empSalary)
    {
        this.empNum=empNum;
        this.empSalary=empSalary;
    }
    void getEmployeeData()
    {
        Scanner sc=new Scanner(System.in);
        System.out.print("Enter the employee id:");
        empNum=sc.nextInt();
        System.out.print("Enter the employee salary:");
        empSalary=sc.nextDouble();
        sc.close();
    }
    public static void main(String[] args)
    {
        employee e1=new employee(0,0);
        e1.getEmployeeData();
        System.out.print(e1.empNum);
        System.out.print(e1.empSalary);
    }  
    }
