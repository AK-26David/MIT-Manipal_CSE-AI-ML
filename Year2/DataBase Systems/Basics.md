# Basics of SQL

#### 1) Creating a table-
```sql
create table employees(
    name			varchar(40),
    dob				date,
    compensation	number(12, 2),
    productivity    number(3, 1)
);
```


#### 2) Adding, modifying, and dropping columns -
```sql
alter table employees add department char(3);
alter table employees modify department varchar(3);
alter table employees drop column department;
```

#### 3) Inserting values -
```sql
insert into employees values('Anon', '01-Jan-1990', 1, 1);
insert into employees(name, compensation) values('John Doe', 9999999);
```

#### 4) Selecting -
```sql
select * from employees;
select compensation as employee_compensation from employees;
select productivity / compensation from employees;
```

#### 5) Updating -
```sql
update employees set productivity=0;
update employees set productivity=1, compensation=1 where name='John Doe';
```

#### 6) Deleting -
```sql
delete from employees where name='Anon';
delete from employees;
```

#### 7) Renaming, truncating, and deleting -
```sql
rename employees to slaves;
truncate table slaves;
drop table slaves;
```