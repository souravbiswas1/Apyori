import numpy as np
import pandas as pd
from apyori import apriori
# from efficient_apriori import apriori
# from mlxtend.frequent_patterns import apriori, association_rules
import os

base_dir = 'D:\\Zencode\\Projects\\POC\\8.Apriori'
os.chdir(base_dir)
sample = pd.read_csv('Data-Fashion.csv',na_values = ['Missing','NA','na','N/A','n/a','','-',' ','undefined','null'],encoding = "ISO-8859-1")
new = sample['timestamp'].str.split(" ", n = 1, expand = True)
sample['Dates'] = new[0]
sample['Time'] = new[1]

mask1 = (sample['Dates'] >= '01-10-2017') & (sample['Dates'] <= '07-10-2017')
mask2 = (sample['Dates'] >= '08-10-2017') & (sample['Dates'] <= '14-10-2017')
mask3 = (sample['Dates'] >= '15-10-2017') & (sample['Dates'] <= '21-10-2017')
df_1 = sample.loc[mask1]
df_2 = sample.loc[mask2]
df_3 = sample.loc[mask3]

product1 = pd.DataFrame(df_1['u2'])
# product1 = product1['u2'].values.tolist()
# product1['u16'] = product1['u16'].astype('str')
df1_product = product1['u2'].str.split(',', expand = True)
# df1_product = product1.values.tolist()


product2 = pd.DataFrame(df_2['u2'])
# product2['u16'] = product2['u16'].astype('str')
df2_product = product2['u2'].str.split(',', expand = True)
# df2_product.to_string(header=False)


product3 = pd.DataFrame(df_3['u2'])
# product3['u16'] = product3['u16'].astype('str')
df3_product = product3['u2'].str.split(',', expand = True)


# efficient apriori--
# transactions1 = []
# for i in range(0, len(df1_product)):
#     transactions1.append([str(df1_product.values[i,j]) for j in range(0,16)])
# itemsets1, rules1 = apriori(df1_product, min_support=0.0006,  min_confidence=0.3)
# print(rules1)
# rules1_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules1)
# for rule1 in sorted(rules1_rhs, key=lambda rule: rule.lift):
# 	print(rule1)


# apyori--
transactions2 = []
for i in range(0, len(df2_product)):
    transactions2.append([str(df2_product.values[i,j]) for j in range(0, 20)])
rules2 = apriori(transactions2, min_support=0.0003, min_confidence=0.2,min_lift = 3,min_length = 2)
l = list(rules2)
print(l)


# rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
# rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

# transactions3 = []
# for i in range(0, len(df3_product)):
#     transactions3.append([str(df3_product.values[i,j]) for j in range(0, 16)])
# itemsets3, rules3 = apriori(transactions3, min_support=0.0009, min_confidence=0.3)
# rules3_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules3)
# for rule3 in sorted(rules3_rhs, key=lambda rule: rule.lift):
# 	print(rule3)
