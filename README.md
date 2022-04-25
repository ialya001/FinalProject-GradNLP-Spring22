# FinalProject-GradNLP-Spring22
# Detect Duplicated Questions in Quora

The folder structure after unzipping is given below:
|-- readme.txt
|-- SemanticSimilarity/
    |-- dataset/
        |-- test.csv
        |-- train.csv
	|-- results/
		|-- screenshots
            |-- LSH Candidates.png
		    |-- Cosine Similarity Candidates.png
		    |-- Needleman Wunch Candidates.png
            |-- result.png
	|-- GradNLP-Final Project-Group5-DetectSemanticSimilarQuestions.py
	|-- GradNLP-Final Project-Group5-DetectSemanticSimilarQuestions.ipnup(for Juypter Notebook version)
    |-- minineedle
        |--__init__.py
		|--__pycache__
		|--core.py
		|--needle.py
		|--tests
 


*Questions Semantic Similarity Detection*
To run Questions Semantic Similarity Detection model, first of all, the required libraries need to be installed. To do so first insure that Python3 is installed in your system along with pip. After moving to the directory 'SemanticSimilarity', then, run the following commands:
run Project-Group5-DetectSemanticSimilarQuestions.py or if you are using  Jupyter Notebook upload the file GradNLP-Final Project-Group5-DetectSemanticSimilarQuestions.ipynp to your Jupyter Notebook browser.

The required library needed to be installed using pip install command:
For example:
pip install [library name]

import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sentence_transformers import SentenceTransformer #Vectorize the sentences using bert
from minineedle import needle, core 



Please change the dataset link to your folder in order to read the dataset i,e:
When you reached to this line:
db = pd.read_csv('/Users/ibrahim/Desktop/CAP5640/FinalProject/Dataset/train.csv')
-----Please change this to your folder location---- to avoid any error



After the required libraries are installed, and the link to your dataset locations is changed the project is ready to run. Then, one can run any one of the models by using the command
>python Project-Group5-DetectSemanticSimilarQuestions.py
Or
>if you are using  Jupyter Notebook upload the file GradNLP-Final Project-Group5-DetectSemanticSimilarQuestions.ipynp to your Jupyter Notebook browser. 


If you would like to test other questions, do the following:
- Go to train dataset and cop any question form question1 columns 
- there is a query variable in the code. Once you spot it please paste your new question 
there and re-run the code.

