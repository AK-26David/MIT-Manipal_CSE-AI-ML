import java.util.Scanner;
class average
{
    double num1;
    double num2;
    double num3;
    void calculate()
    {
        double average=(num1+num2+num3)/3;
        System.out.print("Avergae:"+average);
    }
    public static void main(String[] args)
    {
        Scanner sc=new Scanner(System.in);
        System.out.print("Enter the first number:");
        double num1=sc.nextDouble();
        System.out.print("Enter the second number:");
        double num2=sc.nextDouble();
        System.out.print("Enter the third number:");
        double num3=sc.nextDouble();
        average averagecalculator=new average();
        averagecalculator.num1 = num1;
        averagecalculator.num2 = num2;
        averagecalculator.num3 = num3;
        averagecalculator.calculate();
    }
}