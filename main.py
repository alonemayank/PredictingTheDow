from textblob import TextBlob
import pandas as pd
import numpy as np

#Initialize data
data = pd.read_csv('stocknews/Combined_News_DJIA.csv')
#print(data['News'])

#set count to verify if iteration is successful
count=0
count2=0
#iterating through data
for index, row in data.iterrows():
    top=1
    for tuple in row:
        count2 += 1
        test = TextBlob(str(tuple))
        #calculating polarity of sentiment
        print row['Date'],"Top", top," ", tuple, " : ", test.sentiment.polarity*100
        top += 1
    count += 1
#verifying the iterations
print "Count 1 ",count," Count 2 ",count2

#test
text = '''
Russia ends Georgia operation
'''

blob = TextBlob(text)

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)

