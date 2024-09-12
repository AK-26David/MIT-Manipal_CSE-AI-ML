class A extends Thread
{
    public void run()
    {
        System.out.print(Thread.currentThread().getName());
        System.out.print(Thread.currentThread().getPriority());
    }
}
class priorityTest
{
    public static void main(String[] args)
    {
        A t1= new A();
        A t2= new A();
        A t3= new A();
        t1.setName("t1 thread");
        t2.setName("t2 thread");
        t3.setName("t3 thread");
        t1.setPriority(2);
        t2.setPriority(5);
        t3.setPriority(7 );
        t1.start();
        t2.start();
        t3.start();
    }
}