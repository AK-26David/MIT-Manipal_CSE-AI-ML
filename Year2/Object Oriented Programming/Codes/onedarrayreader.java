import java.util.Scanner;
public class onedarrayreader{
    public static void main(String[] args)
    {
        int n,i;
        Scanner in= new Scanner(System.in);
        System.out.println("Enter the dimensions of the array:");
        n= in.nextInt();
        int a[]=new int[n];
        System.out.println("Enter the elements in the array:");
        for(i=0;i<n;i++)
        {
            a[i]=in.nextInt();
        }
        System.out.println("The printed array is:");
        for(i=0;i<n;i++)
        {
            System.out.println(a[i]);
        }
    }
    }
