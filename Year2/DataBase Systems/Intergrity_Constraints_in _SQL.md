# Integrity Constraints in SQL

### (1) Creating tables with constraints 
**Different Types of Constraints**
1. **NOT NULL** - Coloumn cannot have a null value
2. **DEFAULT** - Provides a default value for the coloumn when not specified
3. **UNIQUE** - All values in the coloumn are different 
4. **PRIMARY** - Uniquely identiffies each row/record in a database table
5. **FORIEGN** - Uniquely identifies row/record in another database table
6. **CHECK** - to check whether the condition is satisfied <br>
**SQL Command to create a table with constraints -**
create table employees(id number(10) primary key,another_id number(10) unique, name varchar(40) not null, department char(3) default('DEF') check(department in('ABC,'DEF)));<br>
     
     create table another_way(id  int references employees(id) on delete cascade);

    create table another_way(id  int references employees(id) on delete cascade);<br>

### (2) Externaly adding named constraints 

alter table slaves add constraint 'constraint_name' check(id!=0);<br>
alter table slaves drop constraint 'constraint_name'<br>

### (3) Externally adding unamed constraints 

alter table slaves modify id not null;

alter table slaves  modify salary default 1000;

### (4) Using joins keyword 

select * from table_1 natural join table_2;
