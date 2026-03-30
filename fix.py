import re 
content = open('src/data_preprocessing.py', encoding='utf-8').read() 
content = content.replace("int(date.month in [11, 12] and date.day >= 20)", "int(pd.Timestamp(date).month in [11, 12] and pd.Timestamp(date).day >= 20)") 
open('src/data_preprocessing.py', 'w', encoding='utf-8').write(content) 
print('Fixed!') 
