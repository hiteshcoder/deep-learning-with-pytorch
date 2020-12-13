# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:32:22 2020

@author: HiteshNayak
"""

#lets say we want to capture information of employees of an office
#we can have it in a class which captures unique features


class employee:
    pass

#each class will have an instance . i.e. the value of an employee

emp1=employee()
emp2=employee()

#Lets see if they have different registors or not
print(emp1)
print(emp2)

#now you might want to give values to the instances

emp1.first='Virgil'
emp1.last='dijk'
emp1.email='virjil.dijk@liverpool.com'
emp1.pay=100

emp2.first='harry'
emp2.last='kane'
emp2.email='harryl.kane@tottenham.com'
emp2.pay=100

emp1.email
emp2.email

#or you can just have a special init method inside of employee class
#this is like a constructor. This will receive the instance automatically (calling itself as self)
class Employee:

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
emp1 = Employee('virgil', 'dijk', 50000)
emp2 = Employee('harry', 'kane', 60000)

emp1.email
emp2.email

#we could also have had the emplyee full name as
print('{} {}'.format(emp1.first, emp1.last))

#or we could have the moethod inside the class as we did above
#each method inside a class takes instance as the first argument as self
#here you use an instance and base a method on it so you dont need to pass argument
#methods take instance automatically as an argument so self in full name
emp1.fullname()
emp2.fullname()

#this is same as passing an class with argument of an instance 
Employee.fullname(emp1)

#Now you can understand that using an instance automatically passes it as an argument called self

#----------------------------------------------------------------------------------------
