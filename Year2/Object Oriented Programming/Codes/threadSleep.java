class A extends Thread
{
    public void run()
    {
        String n=Thread.currentThread().getName();
        try
        {
            for(int i=1;i<=5;i++)
            {
                System.out.print(n);
                Thread.sleep(1000);
            }
        }
        catch(InterruptedException i)
        {
            
        }
    }
}
class threadSleep
{
    public static void main(String[] args)
    {
        A t1= new A();
        A t2= new A();
        A t3= new A();
        t1.setName("thread 1");
        t2.setName("thread 2");
        t3.setName("thread 3");
        t1.start();
        t2.start();
        t3.start();
    }
}