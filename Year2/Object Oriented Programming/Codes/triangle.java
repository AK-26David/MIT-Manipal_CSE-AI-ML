import java.util.Scanner;
class triangle
{
    double base;
    double height;
    double hypotenuse;
    void area_perimeter()
    {
        double area=0.5*base*height;
        double perimeter=base+height+hypotenuse;
        System.out.print("The area is "+area+"and the perimeter is"+perimeter);
    }
    public static void main(String[] args)
    {
        Scanner sc=new Scanner(System.in);
        System.out.print("Enter the base of the triangle:");
        double base=sc.nextDouble();
        System.out.print("Enter the height of the triangle:");
        double height=sc.nextDouble();
        System.out.print("Enter the hypotenuse of the triangle");
        double hypotenuse=sc.nextDouble();
        triangle trianglecalc=new triangle();
        trianglecalc.base=base;
        trianglecalc.height=height;
        trianglecalc.hypotenuse=hypotenuse;
        trianglecalc.area_perimeter();
    }
}