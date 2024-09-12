class Handle
{
    public static void main(String[] args)
    {
        try
        {
            System.out.print("Learn Coding!!");
            int a=20,b=0,c;
            c=a/b;
            System.out.print(c);
            System.out.print("like share");
        }
        catch(ArithmeticException a)
        {
            int x=20,y=0,z;
            z=x/y;
            System.out.print(z);
            System.out.print("Zero Division Error!!");
        }
        finally
        {
            System.out.print("Subscribe");
            System.out.print("Main Method Entered");
        }
    }
}