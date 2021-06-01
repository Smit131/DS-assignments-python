# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:18:38 2021

@author: Sumit
"""

import pandas as pd
import numby as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules # for finding rules


df = pd.read_csv("G:\\Data Science\\Assn 9 - Association rule\\Python code\\book.csv")
df.info()
# Counting the number of 1s or the number of items
df_n = df.sum()
df_n = pd.DataFrame(df_n)
df_n.columns = ["Count"]

# Frequency plot for the data
plt.bar(x = df_n.index,height = df_n.Count)

# NOw first we find rules based on support and confidence

frequent_items = apriori(df,min_support = 0.015,max_len = 8 , use_colnames = True)

# to get the most frequent items 
frequent_items.sort_values("support",ascending = False , inplace=True)

plt.bar(x = list(range(11)),height = frequent_items.support[0:11])

rules_df = association_rules(frequent_items, metric="lift", min_threshold=19)

rules_df.sort_values('lift',ascending = False,inplace=True)
rules_df.head(20)