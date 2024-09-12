class Bus implements Runnable {
    int available = 1, passenger;

    Bus(int passenger) {
        this.passenger = passenger;
    }

    @Override
    public synchronized void run() {
        String name = Thread.currentThread().getName();
        if (available >= passenger) {
            System.out.println(name + " Reserved Seat");
            available=available-passenger;
        } else {
            System.out.println(name + " Seat not available");
        }
    }
}

public class Customer {
    public static void main(String[] args) {
        Bus bus = new Bus(1);
        Thread t1 = new Thread(bus);
        Thread t2 = new Thread(bus);
        Thread t3 = new Thread(bus);

        t1.setName("Arnav");
        t2.setName("Sanjay");
        t3.setName("Karnik");

        t1.start();
        t2.start();
        t3.start();
    }
}
