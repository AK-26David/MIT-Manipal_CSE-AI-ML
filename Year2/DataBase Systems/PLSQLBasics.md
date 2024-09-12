# Rules of PL-SQL Basics

### 1.) Basic Syntax of a PL-SQL Block
        DECLARE
        <declaration>
        BEGIN
        <execution>
        EXCEPTION
        <exception_handling>
        END;
        /


### 2.) Usage of Declaration Block
       DECLARE
       num1 INTEGER;
       num2 REAL;
       num23 DOUBLE PRECISION;

### 3.) User Defined Subtypes
       DECLARE
       SUBTYPE name is varchar(25);
       SUBTYPE message is char(30);
       saluation name; **salutation variable declaration of name type**
       greetings message;
       BEGIN
       salutation:='HELLO';
       greetings:='Welcome to PLSQL';
       dbms_output.put_line(salutation||greetings);
       END;
       /

### 4.) Syntax for variable declaration
        variable_name[constant]datatype[NOT NULL]:=Initial Value

        eg - i) num NUMBER(10,2);
             ii) name varchar(30):='Arnav';

### 5.)Test - 

    DECLARE
    a integer:=10;
    b integer:=20;
    c integer;
    f real;
    BEGIN
    c:=a+b;
    dbms_output.put_line('Value of C is'||c);
    f:=70/30;
    dbms_output.put_line('Value of F is'||f);
    END;
    /                           

    Nested Declare - 

    DECLARE
    BEGIN
    DECLARE
    BEGIN
    END;
    END;
    /

    Assigning values from above table to PL/SQL using the select into clause of SQL - 

    DECLARE
    c_id customers.id%type:=1;
    c_name customers.name%type;
    c_add customers.address%type;
    c_sal customers.salary%type;
    BEGIN
    SELECT name,address,salary into c_name,c_add,c_sal
    FROM Customers
    WHERE id=c_id;
    dbms_output_put_line('Customer'||c_name||'from'||c_addr||'earns'||c_sal);
    END;
    /

    OUTPUT- Customer Rahul from Ahmedabad earns 2000


    Question - Write a PL-SQL Block to calculate the area of a circle:

    DECLARE
    pi constant number:=3.14; cannot change this value
    radius number:=2;
    area number(5,2);
    BEGIN
    area:=pi*radius*radius;
    dbms_output_put_line('AREA IS'||area);
    END;
    /

### 6.) Conditions - 

    IF < condition> THEN
     < action >
     ELSIF <condition> THEN
     < action >
    ELSE
    < action >
    END IF;


    eg - DECLARE 
        grade char(1);
        BEGIN 
        grade:='&g';
        IF grade='A' then
        dbms_output_put_line('Excellent grade');
        ELSIF grade='B' then
        dbms_output_put_line('Very good');
        ELSE
        dbms_output_put_line('Fail');


### 7.) Simple Loop - 
       LOOP
       <Sequence of statements>
       END LOOP;

       DECLARE
       x NUMBER := 0;
       BEGIN
       LOOP
       DBMS_OUTPUT.PUT_LINE ('Inside loop: x = ' ||
       TO_CHAR(x));             
       x := x + 1;
       IF x > 3 THEN EXIT;
       END IF;
       END LOOP;
      -- After EXIT, control resumes here
      DBMS_OUTPUT.PUT_LINE(' After loop: x = ' || TO_CHAR(x));
      END
      /

### 8.) While Loop - 

      WHILE <condition>
      LOOP
      <Action>
      END LOOP;


      DECLARE
      x NUMBER := 0;
      BEGIN
      WHILE x < 4
      LOOP
      DBMS_OUTPUT.PUT_LINE ('Inside loop: x = ' ||
      TO_CHAR(x));
      X := x + 1;
      END LOOP
      END;
      /


### 9.) For Loop - 

      FOR index IN [ REVERSE ] lower_bound..upper_bound LOOP
      statements
      END LOOP ;

      BEGIN
      DBMS_OUTPUT.PUT_LINE ('lower_bound < upper_bound');
      FOR i IN 1..3 LOOP
      DBMS_OUTPUT.PUT_LINE (i);
      END LOOP;
      DBMS_OUTPUT.PUT_LINE ('lower_bound = upper_bound');
      FOR i IN 2..2 LOOP
      DBMS_OUTPUT.PUT_LINE (i);
     END LOOP;
     END;
    /
    DBMS_OUTPUT.PUT_LINE ('lower_bound > upper_bound');
    FOR i IN 3..1 LOOP
    DBMS_OUTPUT.PUT_LINE (i);
    END LOOP;
    END;
    /

### 10.) Exception Handling - 
       EXCEPTION
       WHEN ex_name_1 THEN
       < Exception handler_1>
       WHEN ex_name_2 OR ex_name_3 THEN
       < Exception handler_2>
       WHEN OTHERS THEN
       < Exception handler_3>
       END;

       Declare
       Insufficient_Balance Exception;
       Amount account.balance%Type;
       Temp account.balance%Type;
       ANumber account.account_number%Type;
       BEGIN
       ANumber := &Number;
       Amount := &Amount;
       Select balance into Temp from account where
       account_number = ANumber;
       Temp := Temp – Amount;
       IF (Temp >=500) THEN
       update account set balance = Temp where account_number =
       ANumber;
       ELSE
       RAISE Insufficient_Balance;
       EXCEPTION
       WHEN Insufficient_Balance THEN
       dbms_output.put_line(“Insufficient Balance”);
       WHEN OTHERS THEN
       dbms_output.put_line(“ERROR”);
       END;