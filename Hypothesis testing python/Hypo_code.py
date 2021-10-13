# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:46:29 2021

@author: Sumit
"""
"""
problem statement ->
A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units.
 A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at
 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.

"""

"""
Ho -> there is no difeerence between diameters of two units taken
Ha -> difference is there and action has to be taken
"""
import pandas as pd
import numpy as np
import scipy
from scipy import stats

df = pd.read_csv("G:\Data Science\Assn 3 - Hypothesis Testing\python code\Hypothesis testing python\Cutlets.csv")

print(df.describe())
#all Y is continuos

#1st test check - are all y's normal
print(stats.shapiro(df["Unit A"]))
print(stats.shapiro(df["Unit B"]))

#Ho is both dataset are normal

"""ShapiroResult(statistic=0.9649458527565002, pvalue=0.3199819028377533)
ShapiroResult(statistic=0.9727300405502319, pvalue=0.5224985480308533)"""
#p high null fly -- failed to reject null hypothesis
#All the ys are normal

#Are external conditions same for both batches

#thus we will now perform pairet t test

stats.ttest_rel(df["Unit A"], df["Unit B"])

#Ttest_relResult(statistic=0.7536787225614323, pvalue=0.45623007680384076)
# p high null fly -- failed to reject null
#thus there is no significance between the diamteres in the two batches


"""
Fantaloons Sales managers commented that % of males versus females walking in to the store differ based on day of the week.
 Analyze the data and determine whether there is evidence at 5 % significance level to support this hypothesis.

"""
"""
coding test
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
# can we assume anything from our sample
significance = 0.025
# our samples - 82% are good in one, and ~79% are good in the other
# note - the samples do not need to be the same size
sample_success_a, sample_size_a = (410, 500)
sample_success_b, sample_size_b = (379, 400)
# check our sample against Ho for Ha != Ho
successes = np.array([sample_success_a, sample_success_b])
samples = np.array([sample_size_a, sample_size_b])
# note, no need for a Ho value here - it's derived from the other parameters
stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='two-sided')
# report
print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
if p_value > significance:
   print ("Fail to reject the null hypothesis - we have nothing else to say")
else:
   print ("Reject the null hypothesis - suggest the alternative hypothesis is true")
"""
# here both x and y are discrete

df2 = pd.read_csv("G:\Data Science\Assn 3 - Hypothesis Testing\python code\Hypothesis testing python\Faltoons.csv")

print(df2.describe())

df2.info()
df2["Weekdays"].value_counts()
"""
Female    287
Male      113
Name: Weekdays, dtype: int64
"""

df2["Weekend"].value_counts()
"""
Female    233
Male      167
Name: Weekend, dtype: int64
"""
#importing packages to do 2 proportion test
from statsmodels.stats.proportion import proportions_ztest
females_wday,total_visits_wday = (287,400)
females_wend,total_visits_wend = (233,400)
females_no = np.array([females_wday,females_wend])
total_visits = np.array([total_visits_wday,total_visits_wend])
stat, p_value = proportions_ztest(count=females_no, nobs=total_visits,  alternative='two-sided')

print('{0:0.3f}'.format(p_value))
#p value is lower than 0.005 -- p low null no/go
#null hypothesis is rejected thus the praportions are different
#thus Ha is taken and action is to be taken as amount of female visits differs based on weekday and week end

