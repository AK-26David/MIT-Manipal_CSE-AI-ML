class nestedCatch
{
    public static void main(String[] args)
    {
        try
        {
            System.out.print(10/0);
        }
        catch(Exception e)
        {
            try
            {
                String a=null;
                System.out.print(a.toLowerCase());
            }
            catch(NullPointerException n)
            {
                System.out.print("Null value cannot be converted");
            }
        }
        System.out.print("Main method has ended");
    }
}