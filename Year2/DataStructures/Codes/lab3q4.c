#include <stdio.h>
#include <stdlib.h>

struct DOB {
    int day;
    int *month;
    int year;
};

struct STU_INFO {
    int reg_no;
    char *name;
    char address[50];
};

struct COLLEGE {
    char *college_name;
    char university_name[50];
};

struct STUDENT {
    struct DOB dob;
    struct STU_INFO stu_info;
    struct COLLEGE college;
};

int main() {
    int n;
    printf("Enter the number of students: ");
    scanf("%d", &n);
    struct STUDENT *student = malloc(n * sizeof(struct STUDENT));

    for (int i = 0; i < n; i++) {
        printf("Enter the details of student %d:\n", i + 1);
        printf("----------------------------------\n");
        printf("Enter the date of birth of the student (day month year): ");
        scanf("%d %d %d", &(student[i].dob.day), student[i].dob.month, &(student[i].dob.year));

        student[i].stu_info.name = (char *)malloc(20 * sizeof(char));
        printf("Enter the registration number of the student: ");
        scanf("%d", &(student[i].stu_info.reg_no));

        printf("Enter the name of the student: ");
        scanf("%s", student[i].stu_info.name);

        printf("Enter the address of the student: ");
        scanf("%s", student[i].stu_info.address);

        student[i].college.college_name = (char *)malloc(30 * sizeof(char));
        printf("Enter college name: ");
        scanf("%s", student[i].college.college_name);

        printf("Enter the university name: ");
        scanf("%s", student[i].college.university_name);
    }

    // Printing the entered student details (optional)
    printf("Entered Student Details:\n");
    for (int i = 0; i < n; i++) {
        printf("Student %d:\n", i + 1);
        printf("Date of Birth: %d/%d/%d\n", student[i].dob.day, *(student[i].dob.month), student[i].dob.year);
        printf("Registration Number: %d\n", student[i].stu_info.reg_no);
        printf("Name: %s\n", student[i].stu_info.name);
        printf("Address: %s\n", student[i].stu_info.address);
        printf("College Name: %s\n", student[i].college.college_name);
        printf("University Name: %s\n", student[i].college.university_name);

        // Free allocated memory
        free(student[i].stu_info.name);
        free(student[i].college.college_name);
    }

    free(student); // Free the array of students
    return 0;
}
