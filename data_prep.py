import pandas as pd

df1 = pd.read_excel('above65.xlsx')
df2 = pd.read_excel('hospital_beds.xlsx')
df3 = pd.read_excel('deaths100k.xlsx')

'''
print(df3)
#NaN checking
for i in range(df3.shape[0]):
	if pd.isna(df3.iloc[i,1]):
		print(i)
#features sorted country-wise checking
for i in range(df1.shape[0]):
	if df1.iloc[i,0]!=df2.iloc[i,0]:
		print(i)
'''

dfout = pd.DataFrame(columns=['Country', 'PopulationAbove65', 'HospitalBeds', 'DeathsPer100k'])
outIdx = 0

for i in range(df1.shape[0]):
	val1 = df1.iloc[i, 1]
	val2 = df2.iloc[i, 1]
	if pd.isna(val1) or pd.isna(val2):
		continue
	country = df1.iloc[i, 0]
	val3 = df3[df3['Country'] == country]
	if val3.shape[0] == 0:
		continue
	if pd.isna(val3.iloc[0, 1]):  # only 1 NaN#not necessary
		continue
	dfout.loc[outIdx] = [country, val1, val2, val3.iloc[0, 1]]
	outIdx += 1

dfout.to_excel('data.xlsx', index=False)
