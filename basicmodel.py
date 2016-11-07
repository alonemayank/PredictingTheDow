import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model as LR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def countVectorize(trainheadlines, testheadlines):
    basicvectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), analyzer=stemmed_words)
    # basicvectorizer = CountVectorizer()
    basictrain = basicvectorizer.fit_transform(trainheadlines)
    basictest = basicvectorizer.transform(testheadlines)
    return basictrain, basictest, basicvectorizer

def tdIdfVectorize(trainheadlines, testheadlines):
    td = TfidfVectorizer()
    tdTrain = td.fit_transform(trainheadlines)
    tdTest = td.transform(testheadlines)
    return tdTrain, tdTest, td


def runKNN(basictrain, basictest, train, test):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(basictrain, train["Label"])
    predictions = neigh.predict(basictest)
    matrix = pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
    print("Running KNN gives accuracy of "),
    accuracy(matrix)
    return neigh


def runLogisticReegresion(basictrain, basictest, train, test):
    logModel = LR.LogisticRegression()
    logModel = logModel.fit(basictrain, train["Label"])
    predictions = logModel.predict(basictest)
    matrix = pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
    print("Running Logistical Regression gives accuracy of "),
    accuracy(matrix)
    return logModel


def runLinearSVC(basictrain, basictest, train, test):
    clf = LinearSVC()
    clf.fit(basictrain, train["Label"])
    predictions = clf.predict(basictest)
    matrix = pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
    print("Running LinearSVC gives accuracy of "),
    accuracy(matrix)
    return clf


def runRandomForestClassifier(basictrain, basictest, train, test):
    rfc = RandomForestClassifier()
    rfc.fit(basictrain, train["Label"])
    predictions = rfc.predict(basictest)
    matrix = pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
    print("Running RandomForestClassifier gives accuracy of "),
    accuracy(matrix)
    return rfc

def linearModel(basictrain,basictest, train,test):
	linearModelObj= LR.LinearRegression()
	linearModelObj= linearModelObj.fit(basictrain, train["Open"])
	predictions= linearModelObj.predict(basictest)
	matrix= pandas.crosstab(test["Open"],predictions,rownames=["Actual"], colnames=["Predicted"])
	print ("Running Linear Regression gives accuracy of "),
	accuracy(matrix)
	return linearModelObj


def accuracy(matrix):
    correct = matrix[0][0] + matrix[1][1]
    total = correct + matrix[1][0] + matrix[0][1]
    print(float(correct) * 100 / total)


def CoefToHTML(basicvectorizer, basicmodel, filename):
    basicwords = basicvectorizer.get_feature_names()
    basiccoeffs = basicmodel.coef_.tolist()[0]
    coeffdf = pandas.DataFrame({'Word': basicwords,
                                'Coefficient': basiccoeffs})
    coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    coeffdf.head(10).to_html(filename + '_head.html')
    coeffdf.tail(10).to_html(filename + '_tail.html')


# def main():
# the labels in this file are either 0,1 (will modify accordinly since this not binary classficattion)
data = pandas.read_csv("stocknews/full-table.csv")


# Stop word removal

# for temp in data:

from wordcloud import WordCloud, STOPWORDS
import nltk

from nltk.corpus import stopwords

df = data
print(df.shape)


def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))


import re

# nltk.download()
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
# text = pattern.sub('', text)



from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()

# stop word removal ends

# dividing up the training data per Kaggle instructions, will modify later
train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']




# stem(word) for word in sentence.split(" ")

testheadlines = []
#
for row in range(0, len(test.index)):
    testheadlines.append(''.join(pattern.sub('', str(x).lower())for x in test.iloc[row,2:27]))

trainheadlines = []


for row in range(0, len(train.index)):
    trainheadlines.append(''.join(pattern.sub('', str(x).lower()) for x in train.iloc[row,2:27]))


sent = 'warning with new localization things are being beginning'
# for temp in sent.split(' '): print (st.stem(temp))


cvtrain, cvtest, cvVector = countVectorize(trainheadlines, testheadlines)
tdTrain, tdTest, tdVector = tdIdfVectorize(trainheadlines, testheadlines)

print(data.head(n=10))
print('Running with Count Vectorizer')
logCV = runLogisticReegresion(cvtrain, cvtest, train, test)
runKNN(cvtrain, cvtest, train, test)
runLinearSVC(cvtrain, cvtest, train, test)
runRandomForestClassifier(cvtrain, cvtest, train, test)
linearModel(cvtrain, cvtest, train, test)

print("\nRunning with TD-IDF")
runLogisticReegresion(tdTrain, tdTest, train, test)
runKNN(tdTrain, tdTest, train, test)
svmidf = runLinearSVC(tdTrain, tdTest, train, test)
runRandomForestClassifier(tdTrain, tdTest, train, test)
linearModel(tdTrain, tdTest, train, test)

CoefToHTML(cvVector, logCV, "log-countVector")
CoefToHTML(tdVector, svmidf, "SVM-IDF")




# if __name__ == "__main__":
#	main()