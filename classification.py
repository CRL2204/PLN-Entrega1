import nltk
from nltk.corpus import reuters, stopwords
from nltk.metrics import precision, recall, f_measure
import pickle
import collections
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.stem.snowball import EnglishStemmer
from sklearn.linear_model import LogisticRegression, SGDClassifier


########################################################################################
# DEFININDO AS STOPWORDS


#stemmer
stemmer = EnglishStemmer()

#Stopwords que vamos tirar de cada DOC
stopwords = set(stopwords.words("english"))

#Vou tentar melhorar a lista de stopwords colocando nela algumas pontuacoes q nao servem de nada
# Fiz elas com unicode pq eh assim que as stop_words estao
punctuation = ['.', '-', ',', '"', '(', ')', ':', '?', "'", '--', ';', 
'!', '$', '*', '&', '...', ':/', '/', '%', '..', '1', '2', '3', '4', '5', '6', '7', '8', '9']
punctuation = set(punctuation)

# AGORA AS STOPWORDS VAO TER AS PONTUACOES TB
# PARA FAZER A UNION FOI NECESSARIO TRANSFORMA-LOS EM SET
stopwords = stopwords.union(punctuation)

###########################################################################################


#############################################
#
# VAI CRIAR UMA TUPLA (ARRAY DE FEATURES, CATEGORIA) PARA REPRESENTAR
# CADA UM DOS DOCUMENTOS DE TREINAMENTO
#
#############################################

def create_tuple_words_category(documents):

	categorized_docs = []
	documents_words = []

	for doc in documents:

		#Pegando as palavras do doc que nao sao stopwords para diminuir a quantidade
		doc_words = [stemmer.stem(w) for w in reuters.words(doc) if w not in stopwords]
		documents_words = documents_words + doc_words

		doc_features_category = (doc_words, reuters.categories(doc))
		categorized_docs.append(doc_features_category)

	return categorized_docs, documents_words



categorized_docs = []

# ARMAZENANDO TODOS AS PALAVRAS DOS DOC JA FILTRADOS DOS STOPWORDS
all_docs_words = []


################################################################################
#
# FILTRANDO PARA PEGAR OS ARQUIVOS DE TEST
# COM ELES VAMOS CRIAR TUPLAS (words, categories) PARA DPS O CLASSIFICADOR
# APRENDER A ASSOCIAR AS PALAVRAS DOS DOCS COM AS CLASSES
#
################################################################################


def getting_docs_categorized():
	train_docs = [doc for doc in reuters.fileids() if doc.startswith('training/')]
	print("Training documents: " + str(len(train_docs)))

	test_docs = [doc for doc in reuters.fileids() if doc.startswith('test/')]
	print("Testing documents: " + str(len(test_docs)))

	# ARMAZENANDO AS PALAVRAS DE TODOS OS DOCS DE TREINAMENTO
	all_training_words = []

	# ARMAZENANDO TODOS OS DOCS DE TRAINING COM SUAS CATEGORIAS
	categorized_training_docs = []

	categorized_training_docs, all_training_words = create_tuple_words_category(train_docs)

	# SALVANDO AS TUPLAS EM PICKLE PRA SER USADO DPS SEM PRECISAR TER QUE LER OS DOCUMENTOS
	with open("training_docs2.pickle", "wb") as f:
		pickle.dump(categorized_training_docs, f, 2)

	with open("training_words2.pickle", "wb") as f:
		pickle.dump(all_training_words, f, 2)

	# IMPRIMINDO DADOS DO CONJUNTO DE TREINAMENTO
	print("Training documents words: " + str(len(all_training_words)))
	#print(categorized_training_docs[0])


	# ARMAZENANDO AS PALAVRAS DE TODOS OS DOCS DE TESTE
	all_testing_words = []

	# ARMAZENANDO TODOS OS DOCS DE TRAINING COM SUAS CATEGORIAS
	categorized_testing_docs = []

	categorized_testing_docs, all_testing_words = create_tuple_words_category(test_docs)

	# SALVANDO EM PICKLE PRA SER USADO NO SCRIPT DO DB
	with open("testing_docs2.pickle", "wb") as f:
		pickle.dump(categorized_testing_docs, f, 2)

	with open("testing_words2.pickle", "wb") as f:
		pickle.dump(all_testing_words, f, 2)

	# IMPRIMINDO DADOS DO CONJUNTO DE TREINAMENTO
	print("Testing documents words: " + str(len(all_testing_words)))
	#print(categorized_testing_docs[0])


############################################################################################################

# getting_docs_categorized()


with open("training_docs2.pickle", "rb") as f:
	categorized_training_docs = pickle.load(f)

with open("testing_docs2.pickle", "rb") as f:
	categorized_testing_docs = pickle.load(f)

with open("training_words2.pickle", "rb") as f:
	all_training_words = pickle.load(f)

with open("testing_words2.pickle", "rb") as f:
	all_testing_words = pickle.load(f)

print(len(categorized_training_docs))
print(len(categorized_testing_docs))

all_words = all_training_words + all_testing_words
words_freqDist = nltk.FreqDist(all_words)

print(words_freqDist)
print(words_freqDist.most_common(50))
print(words_freqDist['also'])
print("Quantidade de palavras q so ocorrem uma vez:")
print(len(words_freqDist.hapaxes()))
print("Common words")


# PRA ESCOLHER QUANTAS PALAVRAS VAMOS AVALIAR EM CADA DOCUMENTO
# SE AS 10.000 MAIS COMUNS, POR EXEMPLO
# POIS A 10.000TH palavra mais comum eh ('york', 8) QUE OCORRE 8 VEZES, O QUE PODE SER RELEVANTE
most_common_words_freqDist = words_freqDist.most_common(1000)
#print(most_common_words_freqDist[-2])

# PEGAR SO AS PALAVRAS MAIS COMUNS PARA SERVIREM DE FEATURES
# most_common_words_freqDist RETORNA (word, frequency)
# ENTAO TEMOS Q PEGAR SO AS PALAVRAS:
most_common_words = [w for (w, freq) in most_common_words_freqDist]
print("As palavras mais comuns:")
# print(most_common_words)

# PEGAR TODAS AS PALAVRAS QUE OCORRERAM MAIS DE DUAS VEZES
# meaningful_words_freqDist = [w for w in words_freqDist if words_freqDist[w] > 2]
#print(words_freqDist[-10000])



###########################################################################################

#Retorna uma lista com True ou False dizendo quais palavras da word_features o documento tem
# retorna: {u'even': True, u'story': False, u'also': True, u'see': True, u'much': False,.... }
def find_features(doc_words):
	# Pega todas as palavras do documento e transforma em set pra retornar as palavras independente da frequencia dela
	doc_words = set(doc_words)
	# vai ser o dict dizendo quais palavras, de todas as tidas como mais importantes, estao presentes nese documento
	features = {}
	#print(top_word_features_keys[:20])
	for w in most_common_words:
		features[w] = (w in doc_words)

	return features

#############################################################################################


############################################################################
#
# 1) AGORA TEMOS A QUANTIDADE DE PALAVRAS QUE IREMOS OLHAR EM CADA DOCUMENTO DEFINIDA
# 
# 2) TEMOS O METODO QUE IRA AVALIAR SE AS PALAVRAS MAIS COMUNS ESTAO OU NAO NO DOCUMENTO
#
# 3) AGORA REPRESENTAREMOS OS DOCUMENTOS ATRAVES DAS FEATURES QUE O MESMO TEM OU NAO E DAS CATEGORIAS,
# PARA ISSO CRIAMOS TUPLA ((aval_presenca_das_features), categorias)
#
############################################################################

# CRIANDO DOCUMENTOS COM CATEGORIAS INDIVIDUAIS A PARTIR DOS DOCUMENTOS MULTI-CATEGORIA

training_docs_pre_representation = [ (find_features(doc_words), categories) for (doc_words, categories) in categorized_training_docs]
training_docs_representation = []
for training_doc_representation in training_docs_pre_representation:
	for category in training_doc_representation[1]:
		training_docs_representation.append((training_doc_representation[0], category))


def get_category_docs(docs_representation, wanted_category):
	pos_classified_per_category_docs = []
	neg_classified_per_category_docs = []

	for (doc_features, category) in docs_representation:
		if category == wanted_category:
			pos_classified_per_category_docs.append((doc_features, 'pos'))
		else:
			neg_classified_per_category_docs.append((doc_features, 'neg'))


	pos_len = len(pos_classified_per_category_docs)
	neg_classified_per_category_docs = neg_classified_per_category_docs[:pos_len]

	classified_per_category_docs = pos_classified_per_category_docs + neg_classified_per_category_docs
	random.shuffle(classified_per_category_docs)

	return classified_per_category_docs

main_categories = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

docs_per_category = []

for category in main_categories:
	docs_per_category.append((get_category_docs(training_docs_representation, category), category))

all_classifiers = [ nltk.NaiveBayesClassifier.train(docs_per_category[category][0]) for category in range(10)]

###################################################################################
# IMPRIMINDO A CONTAGEM DE DOCUMENTOS POR CATEGORIA
#
#	for category_docs in docs_per_category:
#		category = category_docs[1]
#		print(category)
#		print(len(category_docs[0]))
###################################################################################

#print(len(training_docs_representation))
#print("Exemplo de um doc representado pelas features e categorias:")
#print(training_docs_representation[9])

testing_docs_pre_representation = [ (find_features(doc_words), categories) for (doc_words, categories) in categorized_testing_docs]
testing_docs_representation = []
for testing_doc_representation in testing_docs_pre_representation:
	for category in testing_doc_representation[1]:
		testing_docs_representation.append((testing_doc_representation[0], category))

testing_docs_per_category = []

for category in main_categories:
	testing_docs_per_category.append((get_category_docs(testing_docs_representation, category), category))



current_category = 0
for classifier in all_classifiers:
	accuracy = nltk.classify.accuracy(classifier, testing_docs_per_category[current_category][0])
	print("accuracy of " + testing_docs_per_category[current_category][1] + " = " + str(accuracy))

	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testing_docs_per_category[current_category][0]):
	    refsets[label].add(i)
	    observed = classifier.classify(feats)
	    testsets[observed].add(i)

	print('pos precision:', precision(refsets['pos'], testsets['pos']))
	print('pos recall:', recall(refsets['pos'], testsets['pos']))
	print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))

	print('neg precision:', precision(refsets['neg'], testsets['neg']))
	print('neg recall:', recall(refsets['neg'], testsets['neg']))
	print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))
	print("\n")

	current_category = current_category + 1

print("lenght of testing docs representations")
print(len(testing_docs_representation))



#############################################################################
# AGORA CLASSIFICAR COM LOGISTIC REGRESSION

print("\n")
print("AGORA ESTAMOS CLASSIFICANDO COM O LOGISTIC REGRESSION!")
print("\n")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
#LogisticRegression_classifier = SklearnClassifier(SGDClassifier())

all_classifiers = [ LogisticRegression_classifier.train(docs_per_category[category][0]) for category in range(10)]

current_category = 0
for classifier in all_classifiers:
	accuracy = nltk.classify.accuracy(classifier, testing_docs_per_category[current_category][0])
	print("Logistic_regression accuracy of " + testing_docs_per_category[current_category][1] + " = " + str(accuracy))

	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testing_docs_per_category[current_category][0]):
	    refsets[label].add(i)
	    observed = classifier.classify(feats)
	    testsets[observed].add(i)

	print('pos precision:', precision(refsets['pos'], testsets['pos']))
	print('pos recall:', recall(refsets['pos'], testsets['pos']))
	print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))

	print('neg precision:', precision(refsets['neg'], testsets['neg']))
	print('neg recall:', recall(refsets['neg'], testsets['neg']))
	print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))

	current_category = current_category + 1

#print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)




########################################################################
#
# AGORA TEMOS TODOS OS DOCUMENTOS REPRESENTADOS ATRAVES DA AVALIACAO DA 
# PRESENCA DAS FEATURES E SUAS CATEGORIAS
#
# COM ISSO, AGORA TREINAREMOS OS CLASSIFICADORES PARA APRENDEREM QUE PALAVRAS
# ESTAO MAIS ASSOCIADAS COM CERTAS CLASSES PARA, ASSIM, ACABAR POR CLASSIFICAR 
# NOVOS CASOS
#
########################################################################

#classifier = nltk.NaiveBayesClassifier.train(training_docs_representation)
#classifier.show_most_informative_features(30)
#accuracy = nltk.classify.accuracy(classifier, testing_docs_representation)
#print("Binary features, common stemmed words, gets %f" % accuracy)








