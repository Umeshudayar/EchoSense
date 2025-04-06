import nltk
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
import logging

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self, method="rake"):
        """Initialize keyword extractor with either RAKE or TF-IDF."""
        self.method = method
        
        if method == "rake":
            self.rake = Rake(stopwords=stopwords.words('english'), punctuations=string.punctuation)
        elif method == "tfidf":
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=10,
                ngram_range=(1, 2),
                max_df=0.85,
                min_df=0.01
            )
        else:
            raise ValueError("Method must be either 'rake' or 'tfidf'")
    
    def extract_rake(self, text):
        """Extract keywords using RAKE algorithm."""
        try:
            self.rake.extract_keywords_from_text(text)
            # Get scores and sort by score (descending)
            keywords_with_scores = self.rake.get_ranked_phrases_with_scores()
            # Format as a list of dictionaries
            result = [
                {"keyword": kw, "score": score}
                for score, kw in keywords_with_scores[:10]  # Limit to top 10
            ]
            return result
        except Exception as e:
            logger.error(f"Error extracting keywords with RAKE: {e}")
            return []
    
    def extract_tfidf(self, text):
        """Extract keywords using TF-IDF."""
        try:
            # For TF-IDF, we need to fit on a corpus, but we only have one document
            # So we'll break it into sentences for a more meaningful analysis
            sentences = nltk.sent_tokenize(text)
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            
            # Get feature names (the words)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get average TF-IDF scores across sentences
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create (word, score) tuples and sort by score
            word_scores = [(word, avg_scores[i]) for i, word in enumerate(feature_names)]
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            
            # Format as a list of dictionaries
            result = [
                {"keyword": word, "score": round(score, 4)}
                for word, score in word_scores[:10]  # Limit to top 10
            ]
            return result
        except Exception as e:
            logger.error(f"Error extracting keywords with TF-IDF: {e}")
            return []
    
    def extract_keywords(self, text):
        """Extract keywords using the selected method."""
        if not text or len(text.strip()) == 0:
            return []
            
        if self.method == "rake":
            return self.extract_rake(text)
        else:
            return self.extract_tfidf(text)