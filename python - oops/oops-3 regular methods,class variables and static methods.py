# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:21:34 2020

@author: HiteshNayak
"""

#you can have a normal method which takes instance as its first argument bydefault 
#or you can have a class method which can take a class argument by just having a @classmethod
#

class Employee:

    num_of_emps = 0
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

        Employee.num_of_emps += 1

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)
#you can see that @classmethod is being used here and cls is taken as the class object
    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


emp_1 = Employee('virjil', 'dijk', 50000)
emp_2 = Employee('robert', 'lewandowski', 60000)

#POINT 1
#--------------------------------------------------------------------------------
#now you can chaange the raise amount by doing this
Employee.set_raise_amt(1.05)
#lets check if it has altered all or not
emp_1.raise_amt
emp_2.raise_amt

#this is same as doing 
Employee.raise_amt=1.06

#this is also same as 
emp_1.set_raise_amt(1.07)
#you can check this by running the 
emp_2.raise_amt
#----------------------------------------------------------------------------------
#POINT 2
#lets say you are getting the class information as a string -( first-last-pay)
#you want to create a instance from this string , you can do it using a class method
#from_string is something that can be used 

emp_str_1='roberto-firmino-2000'
emp_str_1='linoel-messi-4000'
emp_str_3='dele-alli-9000'

new_emp_1=Employee.from_string(emp_str_1)

print(new_emp_1.email)
#-------------------------------------------------------------------------------------
#point3 
#static method
#you use a decorator @staticmethod
#lets say we want to do with the workday of week and identify if its weekday or weekend!
#a method is static if you dont access the instance or class anywhere inside the method
import datetime
my_date = datetime.date(2016, 7, 11)

print(Employee.is_workday(my_date))
#-------------------------------------------------------------------
