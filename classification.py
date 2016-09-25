import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
import pickle


########################################################################################
# DEFININDO AS STOPWORDS


#Stopwords que vamos tirar de cada DOC
stopwords = set(stopwords.words("english"))

#Vou tentar melhorar a lista de stopwords colocando nela algumas pontuacoes q nao servem de nada
# Fiz elas com unicode pq eh assim que as stop_words estao
punctuation = ['.', '-', ',', '"', '(', ')', ':', '?', "'", '--', ';', 
'!', '$', '*', '&', '...', ':/', '/', '%', '..']
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
		doc_words = [w for w in reuters.words(doc) if w not in stopwords]
		documents_words = documents_words + doc_words

		doc_features_category = (doc_words, reuters.categories(doc))
		categorized_docs.append(doc_features_category)

	return categorized_docs, documents_words



categorized_docs = []

# ARMAZENANDO TODOS AS PALAVRAS DOS DOC JA FILTRADOS DOS STOPWORDS
all_docs_words = []


# for doc in reuters.fileids():

# 	#Pegando as palavras do doc que nao sao stopwords para diminuir a quantidade
# 	docs_words = [w for w in reuters.words(doc) if w not in stopwords]
# 	all_docs_words = all_docs_words + docs_words
# 	doc_features_category = (docs_words, reuters.categories(doc))
# 	categorized_docs.append(doc_features_category)

# print(categorized_docs[:5])
# print(len(categorized_docs))
# print(len(all_docs_words))


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
	with open("training_docs.pickle", "wb") as f:
		pickle.dump(categorized_training_docs, f)

	with open("training_words.pickle", "wb") as f:
		pickle.dump(all_training_words, f)

	# IMPRIMINDO DADOS DO CONJUNTO DE TREINAMENTO
	print("Training documents words: " + str(len(all_training_words)))
	#print(categorized_training_docs[0])


	# ARMAZENANDO AS PALAVRAS DE TODOS OS DOCS DE TESTE
	all_testing_words = []

	# ARMAZENANDO TODOS OS DOCS DE TRAINING COM SUAS CATEGORIAS
	categorized_testing_docs = []

	categorized_testing_docs, all_testing_words = create_tuple_words_category(test_docs)

	# SALVANDO EM PICKLE PRA SER USADO NO SCRIPT DO DB
	with open("testing_docs.pickle", "wb") as f:
		pickle.dump(categorized_testing_docs, f)

	with open("testing_words.pickle", "wb") as f:
		pickle.dump(all_testing_words, f)

	# IMPRIMINDO DADOS DO CONJUNTO DE TREINAMENTO
	print("Testing documents words: " + str(len(all_testing_words)))
	#print(categorized_testing_docs[0])


############################################################################################################

#getting_docs_categorized()


with open("training_docs.pickle", "rb") as f:
	categorized_training_docs = pickle.load(f)

with open("testing_docs.pickle", "rb") as f:
	categorized_testing_docs = pickle.load(f)

with open("training_words.pickle", "rb") as f:
	all_training_words = pickle.load(f)

with open("testing_words.pickle", "rb") as f:
	all_testing_words = pickle.load(f)

print(len(categorized_training_docs))
print(len(categorized_testing_docs))

all_words = all_training_words + all_testing_words
words_freqDist = nltk.FreqDist(all_words)
print(words_freqDist)
print(words_freqDist.most_common(50))
print(words_freqDist['also'])
print(len(words_freqDist.hapaxes()))

# common_words = words_freqDist.most_common(15000)
# print(common_words[-1])

# PEGAR TODAS AS PALAVRAS QUE OCORRERAM MAIS DE DUAS VEZES
meaningful_words_freqDist = [w for w in words_freqDist if words_freqDist[w] > 2]
print(len(words_freqDist))
print(len(meaningful_words_freqDist))

###########################################################################################

#Retorna uma lista com True ou False dizendo quais palavras da word_features o documento tem
# retorna: {u'even': True, u'story': False, u'also': True, u'see': True, u'much': False,.... }
def find_features(tweet):
	# Pega todas as palavras do documento e transforma em set pra retornar as palavras independente da frequencia dela
	tweet_words = set(tweet)
	# vai ser o dict dizendo quais palavras, de todas as tidas como mais importantes, estao presentes nese tweet
	features = {}
	#print(top_word_features_keys[:20])
	for w in top_tweets_features:
		features[w] = (w in tweet_words)

	return features