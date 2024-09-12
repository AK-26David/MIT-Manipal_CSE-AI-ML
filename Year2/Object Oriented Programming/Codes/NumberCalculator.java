import java.util.Scanner;
public class NumberCalculator {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Welcome to Arnav Karnik's Calculator program:");
        System.out.println("**********************************************");
        System.out.print("Input the number of integers to be used for calculation: ");
        int n = in.nextInt();
        int[] a = new int[n];
        System.out.println("Enter " + n + " integers:");
        for (int i = 0; i < n; i++) {
            a[i] = in.nextInt();
        }
        System.out.println("Enter the operation which you want to perform:");
        System.out.println("1. Addition");
        System.out.println("2. Subtraction");
        System.out.println("3. Multiplication");
        System.out.println("4. Division");
        System.out.println("5. Modulus");
        System.out.println("6. Percentage");
        System.out.print("Enter your choice: ");
        int choice = in.nextInt();
        switch (choice) {
            case 1:
                add(a, n);
                break;
            case 2:
                subtract(a, n);
                break;
            case 3:
                multiply(a, n);
                break;
            case 4:
                divide(a, n);
                break;
            case 5:
                mod(a, n);
                break;
            case 6:
                per(a, n);
                break;
            default:
                System.out.println("Invalid choice");
                break;
        }
        in.close(); 
    }
    public static void add(int[] a, int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[i];
        }
        System.out.println("The sum is: " + sum);
    }
    public static void subtract(int[] a, int n) {
        int difference = a[0]; 
        for (int i = 1; i < n; i++) {
            difference -= a[i];
        }
        System.out.println("The difference is: " + difference);
    }
    public static void multiply(int[] a, int n) {
        int product = 1;
        for (int i = 0; i < n; i++) {
            product *= a[i];
        }
        System.out.println("The product is: " + product);
    }
    public static void divide(int[] a, int n) {
        if (n == 2) {
            float quotient = (float) a[0] / a[1];
            System.out.println("Division is possible. Quotient: " + quotient);
        } else {
            System.out.println("Division is not possible");
        }
    }
    public static void mod(int[] a, int n) {
        if (n == 2) {
            int modulus = a[0] % a[1];
            System.out.println("Modulus is possible. Remainder: " + modulus);
        } else {
            System.out.println("Modulus is not possible");
        }
    }
    public static void per(int[] a, int n) {
        if (n == 2) {
            float percentage = ((float) a[0] / a[1]) * 100;
            System.out.println("Percentage is possible. Percentage: " + percentage + "%");
        } else {
            System.out.println("Percentage is not possible");
        }
    }
}