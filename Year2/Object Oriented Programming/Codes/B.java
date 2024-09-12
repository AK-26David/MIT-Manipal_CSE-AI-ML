class A extends Thread {
    @Override
    public void run() {
        try
        {
            for(int i=0;i<=5;i++)
            {
                System.out.print("Arnav");
                Thread.sleep(1000);
            }
        }
        catch(InterruptedException i)
        {
            
        }
    }
}
class B
{
    public static void main(String[] args)
    {
        A t=new A();
        t.start();
        try
        {
            for(int i=0;i<=5;i++)
            {
                System.out.print("is awesome!!!");
                Thread.sleep(1000);
            }
        }
        catch(InterruptedException i)
        {
            
        }
    }
}