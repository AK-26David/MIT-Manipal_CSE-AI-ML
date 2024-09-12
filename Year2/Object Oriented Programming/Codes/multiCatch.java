class multiCatch
{
    public static void main(String[] args)
    {
        try
        {
            int a=10,b=2,c;
            c=a/b;
            int arr[]={10,20,30};
            System.out.print(arr[2]);
            String str=null;
            System.out.print(str.toUpperCase());
        }
        catch(Exception x)
        {
            System.out.print("All type exceptions handled");
        }
    }
}