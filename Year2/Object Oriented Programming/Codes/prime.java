import java.util.Scanner;
public class prime
{
    public static void prime(int m, int n)
    {
        for(int i=m;i<=n;i++)
        {
            if(i==1||i==-0)
            {
                continue;
            }
            int flag=1;
            for(int j=2;j<=i/2;j++)
            {
                if(i%j==0)
                {
                    flag=0;
                    break;
                }
            }
            if(flag==1)
            {
                System.out.print(i);
            }
        }
    }
    public static void main(String[] args)
    {
        Scanner sc = new Scanner(System.in);
        int ll,ul;
        System.out.print("Enter the lower limit:");
        ll=sc.nextInt();
        System.out.print("Enter the upper limit:");
        ul=sc.nextInt();
        System.out.println("The prime numbers are:");
        prime(ll,ul);
    }
}