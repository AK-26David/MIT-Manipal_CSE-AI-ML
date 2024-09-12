import java.util.Scanner;
class Student {
    int roll_number;
    int marks;
    Student(int roll_number, int marks) {
        this.roll_number = roll_number;
        this.marks = marks;
    }
    public int getNumber() {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter roll number of the student:");
        int roll_number = sc.nextInt();
        return roll_number;
    }
    public void putNumber() {
        System.out.print("Roll number of the student:" + roll_number);
    }
    public int getMarks() {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter marks of the student:");
        int marks = sc.nextInt();
        return marks;
    }          
    public void putMarks() {
        System.out.print("Marks of the student:" + marks);
    }
}
interface Sports {
    void putGrade(String grade);
}
class Result extends Student implements Sports {
    Result(int roll_number, int marks) {
        super(roll_number, marks);
    }

    public void putGrade(String grade) {
        System.out.print("The grade obtained by the student is:" + grade);
    }
}
