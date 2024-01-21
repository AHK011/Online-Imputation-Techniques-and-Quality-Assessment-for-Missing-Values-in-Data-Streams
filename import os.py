import os
import PyPDF2
import nltk
from gensim import corpora, models
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

# Function for lemmatization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]


# more stop words
stop_words = set(stopwords.words('english'))

custom_stop_words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
              ,'b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
              , 'er', '𝑟𝑗', '𝑟𝑖', 'id', '𝑚', 'ik', 'feature', 'mi', 'xo', 'xi', 'si', 'al', 'et', '31', 'kzn', 'kw', 'tzr', 'ty', 'tz', 'yt', 'oz', 'wk', 'wo'] 
stop_words.update(custom_stop_words)



folder_path = r"\Users\kkhha\OneDrive\Desktop\topicModeling\pdf\pp"

documents = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Tokenize and remove stop words
            words = re.findall(r'\b\w+\b', text.lower())
            filtered_words = [word for word in words if word not in stop_words]

            # Lemmatization
            lemmatized_words = lemmatize_text(filtered_words)

            documents.append(lemmatized_words)
    except Exception as e:
        print(f"Error in processing: {file_path}")
        print(f"Error details: {e}")


dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20, alpha= 0.01, eta= 0.01)

num_topics = 5
for topic_id in range(num_topics):
    print(f"Topic {topic_id+1}")
    topic_words = lda_model.show_topic(topic_id, topn=7)
    for word, weight in topic_words:
        print(f"Word: {word}, Weight: {weight}")
    print()
