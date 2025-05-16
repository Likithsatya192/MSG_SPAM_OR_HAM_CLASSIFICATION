# Message Spam or Ham Classifier

This project is a machine learning solution for classifying SMS messages as either **spam** or **ham** (not spam). It uses natural language processing (NLP) techniques and a Random Forest classifier to detect spam messages based on their content.

## Features & Approach
- Text preprocessing: cleaning, tokenization, lemmatization, and stopword removal.
- Word embeddings: Trained using Gensim's Word2Vec on the SMS corpus.
- Feature extraction: Each message is represented by the average of its word vectors.
- Classification: RandomForestClassifier from scikit-learn.

## Environment Setup
This project uses **conda** for environment and dependency management.

### 1. Clone the repository
```bash
git clone <repo-url>
cd MSG_SPAM_OR_HAM_CLASSIFICATION
```

### 2. Create and activate the conda environment
```bash
conda create -p venv python=3.10 -y
conda activate venv/
```

### 3. Install required libraries
Install the dependencies using conda and pip:
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
The notebook will automatically download required NLTK data (stopwords, wordnet) on first run. If you want to do it manually:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage
1. **Place the dataset**: Ensure `data/SMSSpamCollection.txt` is present in the correct location.
2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
3. **Open and run** `spam_or_ham.ipynb` cell by cell.

## Project Structure
```
MSG_SPAM_OR_HAM/
├── data/
│   └── SMSSpamCollection.txt
├── spam_or_ham.ipynb
├── README.md
└── requirements.txt
```

## Main Libraries Used
- pandas
- numpy
- scikit-learn
- gensim
- nltk
- tqdm
- ipykernel

## Results
- The model achieves high accuracy (over 97%) on the test set.
- Evaluation metrics such as confusion matrix and classification report are displayed at the end of the notebook.

## License
This project is for educational purposes. Please check the dataset's license for usage restrictions.

---
Feel free to contribute or raise issues! 