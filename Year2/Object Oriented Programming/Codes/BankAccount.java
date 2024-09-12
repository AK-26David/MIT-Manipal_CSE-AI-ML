import java.util.Scanner;

public class BankAccount {
    // Static data member
    private static double rateOfInterest = 4.0; // Example interest rate (you can set your own value)

    // Data members
    private String name;
    private String accountNumber;
    private String accountType;
    private double balance;

    // Default constructor
    public BankAccount() {
        name = "";
        accountNumber = "";
        accountType = "";
        balance = 0.0;
    }

    // Parameterized constructor
    public BankAccount(String name, String accountNumber, String accountType, double balance) {
        this.name = name;
        this.accountNumber = accountNumber;
        this.accountType = accountType;
        this.balance = balance;
    }

    // Method to deposit amount
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Amount " + amount + " deposited successfully.");
        } else {
            System.out.println("Invalid deposit amount. Amount must be greater than 0.");
        }
    }

    // Method to withdraw amount after checking for minimum balance
    public void withdraw(double amount) {
        if (amount > 0) {
            if (balance - amount >= 500.0) { // Assuming a minimum balance of 500
                balance -= amount;
                System.out.println("Amount " + amount + " withdrawn successfully.");
            } else {
                System.out.println("Insufficient balance to withdraw this amount.");
            }
        } else {
            System.out.println("Invalid withdrawal amount. Amount must be greater than 0.");
        }
    }

    // Method to display all details of an account holder
    public void displayAccountDetails() {
        System.out.println("Account Holder Name: " + name);
        System.out.println("Account Number: " + accountNumber);
        System.out.println("Account Type: " + accountType);
        System.out.println("Account Balance: " + balance);
        System.out.println("Rate of Interest: " + rateOfInterest + "%");
    }

    // Static method to display the rate of interest
    public static void displayInterestRate() {
        System.out.println("Current Rate of Interest: " + rateOfInterest + "%");
    }

    public static void main(String[] args) {
        // Create a Scanner object for user input
        Scanner scanner = new Scanner(System.in);

        // Take user input to create a bank account
        System.out.println("Enter the account holder's name:");
        String name = scanner.nextLine();

        System.out.println("Enter the account number:");
        String accountNumber = scanner.nextLine();

        System.out.println("Enter the account type:");
        String accountType = scanner.nextLine();

        System.out.println("Enter the initial balance:");
        double balance = scanner.nextDouble();

        // Create a bank account object
        BankAccount account1 = new BankAccount(name, accountNumber, accountType, balance);

        // Display account details
        account1.displayAccountDetails();

        // Take user input to perform deposit
        System.out.println("Enter the amount to deposit:");
        double depositAmount = scanner.nextDouble();
        account1.deposit(depositAmount);

        // Take user input to perform withdrawal
        System.out.println("Enter the amount to withdraw:");
        double withdrawalAmount = scanner.nextDouble();
        account1.withdraw(withdrawalAmount);

        // Display updated account details
        account1.displayAccountDetails();

        // Display the rate of interest
        BankAccount.displayInterestRate();

        // Close the Scanner
        scanner.close();
    }
}
