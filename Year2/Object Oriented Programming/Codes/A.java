class A
{
    public static void main(String[] args)
    {
        int a=10,b=0,c;
        try
        {
            c=a/b;
            System.out.print(c);
        }
        catch(Exception e)
        {
            System.out.print(e);
        }
    }
}