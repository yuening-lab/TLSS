!pip install -U sentence-transformers
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
import logging
import os
import gzip
import numpy as np
import torch
from typing import Union, Tuple, List, Iterable, Dict
import pandas as pd
# Download Dense file in SBERT website
from Dense import *

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model.max_seq_length = 500

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences, convert_to_numpy=True)
pca = PCA(n_components=50)
pca.fit(embeddings)
pca_comp = np.asarray(pca.components_)

# Add a dense layer to the model, so that it will produce directly embeddings with the new size
dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=50, bias=False, activation_function=torch.nn.Identity())
dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
model.add_module('dense', dense)

model.save('models/my-500dim-model')
model = SentenceTransformer('models/my-500dim-model')

# Train Sentence Transformer Model and Save the Results
model_train = SentenceTransformer('models/my-500dim-model')
embeddings = model_train.encode(sentences, convert_to_numpy=True)
