class A extends Thread
{
    public void run()
    {
        String name = Thread.currentThread().getName();
        for(int i=1;i<=3;i++)
        {
            System.out.print(name);
        }
    }
}
class stopTest
{
    public static void main(String[] args)
    {
        A t1= new A();
        A t2= new A();
        A t3= new A();
        t1.setName("t1 thread");
        t2.setName("t2 thread");
        t3.setName("t3 thread");
        t1.start();
        t2.start();
        t3.start();
        t2.stop();
    }
}