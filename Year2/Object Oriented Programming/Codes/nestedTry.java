class nestedTry
{
    public static void main(String[] args)
    {
        try
        {
            try
            {
                int a[]={10,20,30};
                System.out.print(a[5]);
            }
            catch(ArrayIndexOutOfBoundsException a)
            {
                System.out.print(a);
            }
            System.out.print(10/0);
        }
        catch(ArithmeticException e)
        {
            System.out.print(e);
        }
    }
}