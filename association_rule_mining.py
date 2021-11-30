#-------------------------------------------------------------------------
# AUTHOR: Nate Colbert
# FILENAME: association_rule_mining.py
# SPECIFICATION: Implements the association rule mining algorithm with a test data set.
# FOR: CS 4200- Assignment #5
# TIME SPENT: 5-7 hours
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
count = [0,0,0,0,0,0,0,0,0]
for index, row in df.iterrows():
    vals = [0,0,0,0,0,0,0,0,0]
    for i in range(7):
        if row[i] == 'Bread':
            vals[0] = 1
            count[0] += 1
        if row[i] == 'Cheese':
            vals[1] = 1
            count[1] += 1
        if row[i] == 'Meat':
            vals[2] = 1
            count[2] += 1
        if row[i] == 'Diaper':
            vals[3] = 1
            count[3] += 1
        if row[i] == 'Wine':
            vals[4] = 1
            count[4] += 1
        if row[i] == 'Milk':
            vals[5] = 1
            count[5] += 1
        if row[i] == 'Pencil':
            vals[6] = 1
            count[6] += 1
        if row[i] == 'Bagel':
            vals[7] = 1
            count[7] += 1
        if row[i] == 'Eggs':
            vals[8] = 1
            count[8] += 1


    labels = {'Bread' : vals[0], 'Cheese' : vals[1], 'Meat' : vals[2], 'Diaper' : vals[3], 'Wine' : vals[4],
             'Milk' : vals[5], 'Pencil' : vals[6], 'Bagel' : vals[7], 'Eggs' : vals[8]}

    encoded_vals.append(labels)

#print(encoded_vals[0])

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below

for index, row in rules.iterrows():

    suportCount = 0

    for x in row['antecedents']:
        print(x, end = "")
        print(', ', end = "")
    for x in row['consequents']:
        print("->", x)
        if x == 'Bread':
            suportCount += count[0]
        if x == 'Cheese':
            suportCount += count[1]
        if x == 'Meat':
            suportCount += count[2]
        if x == 'Diaper':
            suportCount += count[3]
        if x == 'Wine':
            suportCount += count[4]
        if x == 'Milk':
            suportCount += count[5]
        if x == 'Pencil':
            suportCount += count[6]
        if x == 'Bagel':
            suportCount += count[7]
        if x == 'Eggs':
            suportCount += count[8]

    print("Support: ", row['support'])

    print("Confidence: ", row['confidence'])
   


#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below
     

    rule_confidence = row['confidence']

    prior = suportCount/len(encoded_vals)
    print("prior = ", prior)
    print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior), '\n')
  

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()