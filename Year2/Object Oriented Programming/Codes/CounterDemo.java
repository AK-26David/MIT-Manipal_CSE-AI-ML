//Write a Java program to create a Counter class with a private count instance variable and two 
//methods. The first method increment( ) tries to increments count by 1. If count is already at its maximum 
//of 3, then it waits until count is less than 3 before incrementing it. The other method decrement( ) 
//attempts to decrement count by 1. If count is already at its minimum of 0, then it waits until count is 
//greater than 0 before decrementing it. Every time either method has to wait, it displays a statement saying 
//why it is waiting. Also, every time an increment or decrement occurs, the counter displays a statement 
//that says what occurred and shows count's new value. 
//Demonstrates how two synchronized methods need to communicate to keep a counter between 0 and 3 
//when one thread repeatedly tries to increase the counter and the other thread repeatedly tries to 
//decrease the counter. One thread class calls the Counter's increment( ) method 20 times. In between each 
//call, it sleeps for a random amount of time between 0 and 500 milliseconds and the other thread calls the 
//Counter's decrement( ) method 20 times. In between each call, it sleeps for a random amount of time 
//between 0 and 500 milliseconds. 
//Write a CounterDemo class with a main( ) method that creates one Counter and the two threads and 
//starts the threads running. Create just one thread class by implementing Runnable interface that either 
//increments or decrements the counter, depending on a parameter passed to the thread class's 
//constructor . Make sure that the main thread ends at the last. //
class Counter
{
    private int count=0;
    public synchronized void increment()
    {
        while(count==3)
        {
            try
            {
                System.out.print("Waiting to increment!!! Count is already at max");
                wait();
            }
            catch(InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }
        }
        count++;
        System.out.print("Incremented!!! New count:"+count);
        notifyAll();
    }
    public synchronized void decrement()
    {
        while(count==0)
        {
            try
            {
                System.out.print("Waiting to decrement!!! Count is already at min");
                wait();
            }
            catch(InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }
        }
        count--;
        System.out.print("Decremented!!!! New count:"+count);
        notifyAll();
    }
}
class CounterThread implements Runnable
{
    private Counter counter;
    private boolean increment;
    public CounterThread(Counter counter, boolean increment)
    {
        this.counter=counter;
        this.increment=increment;
    }
    @Override
    public void run()
    {
        for(int i=0;i<20;i++)
        {
            if(increment)
            {
                counter.increment();
            }
            else
            {
                counter.decrement();
            }
            try
            {
                Thread.sleep((int)(Math.random()*500));
            }
            catch(InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }
        }
    }
}
public class CounterDemo
{
    public static void main(String[] args)
    {
        Counter counter = new Counter();
        Thread incrementThread= new Thread(new CounterThread(counter,true));
        Thread decrementThread= new Thread(new CounterThread(counter,false));
        incrementThread.start();
        decrementThread.start();
        try
        {
            incrementThread.join();
            decrementThread.join();
        }
        catch(InterruptedException e)
        {
            e.printStackTrace();
        }
        System.out.print("Main thread ends");
    }
}