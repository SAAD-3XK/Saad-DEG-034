#importing dataclass
from dataclasses import dataclass

#decorator
@dataclass
class Mountain:
	#initializing
	name: str
	elevation: float

#creating first object	
mountain1=Mountain("Mt. Everest", 8499)
print(type(mountain1))

#converting the object to string type
mountain1 = str(mountain1)
print(type(mountain1))
