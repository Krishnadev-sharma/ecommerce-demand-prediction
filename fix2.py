content = open('src/data_preprocessing.py', encoding='utf-8').read() 
content = content.replace("season_boost = 20 if date.month in [11, 12] else (10 if date.month in [6, 7] else 0)", "date_ts = pd.Timestamp(date); season_boost = 20 if date_ts.month in [11, 12] else (10 if date_ts.month in [6, 7] else 0)") 
open('src/data_preprocessing.py', 'w', encoding='utf-8').write(content) 
print('Fixed 2!') 
