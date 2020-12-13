# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 08:51:59 2020

@author: HiteshNayak
"""
#class instances are something unique to each instance but class variables are common to all
#lets say about the pay hike 

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
        #you could have had int(self.pay * 1.04) but this wont have let you change the pay hike later for each instance
        self.pay = int(self.pay * self.raise_amt)
    
    
#the class variable isnt available with the instance
print(emp1.__dict__)
#but in the employee class
print(Employee.__dict__)  

#you can change the raise_amount of each instance 
emp1.raise_amt=1.05

print(Employee.raise_amt)
print(emp1.raise_amt)
print(emp2.raise_amt)

#now think of a case where you dont have to change the variable for each instance
#while also having it changed with every time you crrate an instance. Lets say no.of emp
#you can see that they change to 2
print(Employee.num_of_emps)

#so there is a difference between instance variables like self.pay or self.last
#and class variable like raise_amt


