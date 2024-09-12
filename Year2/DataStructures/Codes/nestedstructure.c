#include <stdio.h>

// Define the structure for date of birth (dob) outside of the student structure
typedef struct {
    int day;
    int month;
    int year;
} dob;

// Define the student structure
typedef struct {
    int id;
    char name[20];
    dob date_of_birth; // Use the previously defined 'dob' structure here
} student;

int main() {
    int n;

    printf("Enter the number of students you wish to enter: ");
    scanf("%d", &n);

    // Declare an array of student structures
    student students[n];

    // Input student details
    for (int i = 0; i < n; i++) {
        printf("\nEnter details for student %d:\n", i + 1);
        printf("Enter name of student: ");
        scanf("%s", students[i].name);
        printf("Enter student id: ");
        scanf("%d", &students[i].id);
        printf("Enter the date of birth of the student (day month year): ");
        scanf("%d %d %d", &students[i].date_of_birth.day, &students[i].date_of_birth.month, &students[i].date_of_birth.year);
    }

    // Display student records
    printf("\nBelow are the records displayed:\n");
    for (int i = 0; i < n; i++) {
        printf("\nStudent %d:\n", i + 1);
        printf("Name: %s\n", students[i].name);
        printf("ID: %d\n", students[i].id);
        printf("Date of birth: %d/%d/%d\n", students[i].date_of_birth.day, students[i].date_of_birth.month, students[i].date_of_birth.year);
    }

    return 0;
}
