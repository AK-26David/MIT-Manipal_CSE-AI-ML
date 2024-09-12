import java.util.Scanner;

class Member {
    String Name;
    int age;
    int phone_number;
    String address;
    int Salary;

    Member(String Name, int age, int phone_number, String address, int Salary) {
        this.Name = Name;
        this.age = age;
        this.phone_number = phone_number;
        this.address = address;
        this.Salary = Salary;
    }

    void printSalary() {
        System.out.println("Salary of the employee is: " + Salary);
    }
}

class Employee extends Member {
    String specialization;

    Employee(String Name, int age, int phone_number, String address, int Salary, String specialization) {
        super(Name, age, phone_number, address, Salary);
        this.specialization = specialization;
    }

    void printSpecialization() {
        System.out.println("Employee specialization is: " + specialization);
    }
}

public class Test {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the name of the employee:");
        String Name = sc.nextLine();
        System.out.print("Enter the age of the employee:");
        int age = sc.nextInt();
        sc.nextLine(); // Consume the newline character
        System.out.print("Enter the phone number of the employee:");
        int phone_number = sc.nextInt();
        sc.nextLine(); // Consume the newline character
        System.out.print("Enter the address of the employee:");
        String address = sc.nextLine();
        System.out.print("Enter the salary of the employee:");
        int Salary = sc.nextInt();
        sc.nextLine(); // Consume the newline character
        System.out.print("Enter the specialization of the employee:");
        String specialization = sc.nextLine();

        Employee e1 = new Employee(Name, age, phone_number, address, Salary, specialization);
        e1.printSalary();
        e1.printSpecialization();
        sc.close();
    }
}
