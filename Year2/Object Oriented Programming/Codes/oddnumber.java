import java.util.Scanner;
public class oddnumber
{
    public static void odd(int num)
    {
        if (num%2!=0)
        {
           System.out.print(num + " is an odd number");
        }
        else 
        {
            System.out.print(num + " is not an odd number");
        }
    }
    public static void main(String[] args)
    {
        Scanner sc= new Scanner(System.in);
        int num;
        System.out.print("Enter a number:");
        num=sc.nextInt();
        odd(num);
    }
}