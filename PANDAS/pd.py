import pandas as pd
import numpy as np

country=['india','china','america','geogria','australia','germany']
print(pd.Series(country))

runs=[10,29,37,56,38,0,1,2,33,3]
print(pd.Series(runs))

marks=[29,32,98,89,82]
subject=['maths','science','english','hindi','chinese']
print(pd.Series(marks,index=subject,name='shrishti s marks'))