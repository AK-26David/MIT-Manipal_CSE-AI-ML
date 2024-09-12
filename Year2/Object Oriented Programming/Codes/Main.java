class Table {
    // Synchronized method to print the table
    public synchronized void printTable(int n) {
        for (int i = 1; i <= 10; i++) {
            System.out.print(n * i + " ");
        }
        System.out.println(); // Print a newline after each table
    }
}

class Thread1 extends Thread {
    Table t;

    Thread1(Table t) {
        this.t = t;
    }

    public void run() {
        t.printTable(5);
    }
}

class Thread2 extends Thread {
    Table t;

    Thread2(Table t) {
        this.t = t;
    }

    public void run() {
        t.printTable(10);
    }
}

public class Main {
    public static void main(String[] args) {
        Table obj = new Table();
        Thread1 t1 = new Thread1(obj);
        Thread2 t2 = new Thread2(obj);

        t1.start();
        t2.start();
    }
}
