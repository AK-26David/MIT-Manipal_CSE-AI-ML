import java.util.Scanner;

class Rectangle {
    int length;
    int width;

    Rectangle(int length, int width) {
        this.length = length;
        this.width = width;
    }

    void perimeter() {
        int p;
        p = 2 * (length + width);
        System.out.println("Perimeter:" + p);
    }

    void area() {
        int a;
        a = length * width;
        System.out.println("Area:" + a);
    }
}

class Square extends Rectangle {
    Square(int side) {
        super(side, side);
    }
}

public class Test1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the length of the rectangle:");
        int length = sc.nextInt();
        System.out.print("Enter the width of the rectangle:");
        int width = sc.nextInt();
        System.out.print("Enter the side of the square:");
        int side = sc.nextInt();

        Rectangle r = new Rectangle(length, width);
        System.out.println("\nRectangle:");
        r.perimeter();
        r.area();

        Square sq = new Square(side);
        System.out.println("\nSquare:");
        sq.perimeter();
        sq.area();

        sc.close();
    }
}
