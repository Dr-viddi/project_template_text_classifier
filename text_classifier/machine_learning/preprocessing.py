"""Collection of text preprocessing functions."""
from keras.preprocessing.text import Tokenizer
import numpy
import pandas
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopwords_english = stopwords.words('english')
STOPWORDS = set(stopwords.words('english'))


class Preprocessor:
    """Preprocessor class for preprocessing data.
    """

    @staticmethod
    def remove_non_alphabetic_symbols(text: str) -> str:
        """Delete bad symbols in text string.

        Args:
            text: Input textstring possibly containing bad symbols.

        Returns:
            Processed textstring without bad symbols.
        """
        alphabetic_symbols = re.compile('[^a-zA-Z]')
        return alphabetic_symbols.sub(' ', text)

    @staticmethod
    def remove_multiple_spaces(text: str) -> str:
        """Remove multiple spaces from a test string.

        Args:
            text: Input textstring possibly containing multiple whitespaces.

        Returns:
            Processed textstring without multiple whitespaces.
        """
        return re.sub(' +', ' ', text)

    @staticmethod
    def remove_single_characters(text: str) -> str:
        """Remove single characters from a test string.

        Args:
            text: Input textstring possibly containing single characters.

        Returns:
            Processed textstring without single characters.
        """
        return re.sub(r"\s+[a-zA-Z]\s+", '', text)

    @staticmethod
    def lowercase_text(text: str) -> str:
        """Lowercase the text.

        Args:
            text: Input textstring possibly upper case symbols.

        Returns:
            Processed textstring with only lowercase.
        """
        return text.lower()

    @staticmethod
    def remove_stopwords(text: str, stopwords: set = STOPWORDS) -> str:
        """Remove stopwords from text string.

        Args:
            text: Text string that possibily contains stopwords.
            stopwords: List of stopwords to be removed. Default english
                stopword set from NLTK library.

        Returns:
            Text without stopwords.
        """
        return ' '.join(word for word in text.split() if word not in stopwords)

    @staticmethod
    def stem_words(text: str, stemmer: PorterStemmer = PorterStemmer()) -> str:
        """Reduce a inflected or derived words to its word stem,
            base or root form.

        Args:
            text: Text with inflected or derived words.
            stemmer: A word stemmer. Default is PorterStemmer.

        Returns:
            text with stemmed word
        """
        words = text.split()
        stemmed_words = []
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text

    @staticmethod
    def Vectorizer(data: pandas.core.series.Series) -> list[numpy.ndarray,
                                                            TfidfVectorizer]:
        """Vectorize the training and testing data.

        Args:
            data: Data to be vectorized

        Returns:
            A list of the vectorized training data and fitted vectorizer.
        """
        tfidf = TfidfVectorizer(sublinear_tf=True,
                                min_df=5,
                                ngram_range=(1, 2),
                                stop_words='english'
                                )
        fitted_vectorizer = tfidf.fit(data)
        vectorized_features = fitted_vectorizer.transform(data).toarray()
        return vectorized_features, fitted_vectorizer

    @staticmethod
    def tokenize(data: pandas.core.series.Series,
                 max_vocab_size: int = None,
                 oov_token: str = "<OOV>"
                 ) -> list[numpy.ndarray, Tokenizer, int]:
        """Tokenize the training and testing data.

        Args:
            data: Data to be tokenized.
            max_vocab_size (optional): The maximum number of words to
                keep. Defaults to None.
            oov_tok (optional): Out-of-vocabulary word for uknown words.
                Defaults to "<OOV>".

        Returns:
            A list of the tokenized training data, fitted vectorizer, and
                the acutal vocabulary size.
        """
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        training_sequences = tokenizer.texts_to_sequences(data)
        vocab_size = len(tokenizer.word_index) + 1
        return training_sequences, tokenizer, vocab_size

    @staticmethod
    def cut_text(text: str, max_length: int = 1000) -> str:
        """Reduces the length of a text in which only the first max_length
            characters are kept.

        Args:
            text: Long input text.
            max_length (optional): Number of characters that are be kept.
                Defaults to 1000.

        Returns:
            Shortend text.
        """
        if max_length > len(text):
            return text[0:max_length]
        else:
            return text

    @staticmethod
    def encode_labels(labels: numpy.ndarray) -> numpy.ndarray:
        """Encodes the labels as integers.

        Args:
            labels: Numpy array of labels.

        Returns:
            Numpy array of encoded labels
        """
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(labels)
