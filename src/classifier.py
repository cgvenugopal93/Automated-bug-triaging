from pandas import DataFrame
import numpy
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from sklearn.externals import joblib


class Classifier:
	def __init__(self, tossing_graph, developers, preprocessor):
		self.tossing_graph = tossing_graph
		self.developers = developers
		self.preprocessor = preprocessor

	def run(self, desc):
		data = joblib.load('OutputFiles/data.pkl')

		df_new1, df_new2 = data[13344:], data[:13344]
		print "Training dataset :", len(df_new1)
		print "Testing Dataset :", len(df_new2)

		# Using pipeline and no bigram
		pipeline = Pipeline([
			('vectorizer', CountVectorizer()),
			('classifier', MultinomialNB())])
		#print df_new1['text']
		#print df_new1['class']
		pipeline.fit(numpy.asarray(df_new1['text']), numpy.asarray(df_new1['class']))
		# predicted=pipeline.predict(df_new2['text'])
		print "The accuracy for MultinomialNB is : ", pipeline.score(numpy.asarray(df_new2['text']),
																	numpy.asarray(df_new2['class']))

		# Give input here
		print ("Enter bug description : ")
		#examples = [self.preprocessor.stem_and_stop(raw_input())]
		examples = [self.preprocessor.stem_and_stop(desc)]
		predictions = pipeline.predict(examples)
		# predictions = svm_clf.predict(examples)
		predictions.astype(int)
		print ("The predicted developer is : " + self.developers[int(predictions[0])])  # [1, 0]
		self.tossing_graph.calculate_toss_possibility(self.developers[int(predictions[0])])
		return self.developers[int(predictions[0])]
