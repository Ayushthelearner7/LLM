from pydantic import BaseModel ,EmailStr,Field
from typing import Optional

class Student (BaseModel):

    name : str 
    age : Optional[int] = None
    email : EmailStr
    cgpa: float = Field(gt=0 ,lt=10)



new_student = {'name' : 'ayush' , 'age':32 , 'email':'abc@gmail.com', 'cgpa':6.88}
student = Student(**new_student)

print(student)