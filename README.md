We show that the use of pre-trained embeddings or
pre-trained architectures provides a solid basis for achieving
high results on the identification of hatefull tweets. The RoBERTa-based model achieves better results than the CNN
model. The increase in performance is due to the pre-trained
embeddings provided by a model trained on tweets.


See report.pdf for more information

### Run:
- download dataset at https://bit.ly/3HUgNPo and extract into the folder `raw_data`
- run `run.py` from the main project folder

### Installation:
- install pytorch: https://pytorch.org/get-started/locally/
- run `pip install torchtext torchmetrics mlxtend matplotlib install nltk emoji transformers regex requests hydra-core omegaconf statsmodels seaborn scipy`
- install spaCy: https://spacy.io/usage


