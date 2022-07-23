#ifndef EMPLOYEE_H
#define EMPLOYEE_H

class Employee {
public:
	Employee(const char* name);
	virtual ~Employee();

	virtual void print_description() const = 0;	//	Declaring this abstract since it doesn't print endl.
	void print_salary() const;

protected:
	virtual int salary() const = 0;

private:
	char* m_name;
};

#endif