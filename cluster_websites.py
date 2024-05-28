import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Script started")

# Data inladen
logging.info("Loading dataset")
data = pd.read_csv("analyzed_data.csv")

texts = data['scraped_text_for_sustainability'].tolist()
logging.info(f"Loaded {len(texts)} texts")

# all-MiniLM-L6-v2 inladen
logging.info("Loading sentence transformer model")
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(texts)
logging.info("Model loaded and sentences encoded")

# K-means clustering
kmeans = KMeans(n_clusters=8)
kmeans.fit(sentence_embeddings)
cluster_labels = kmeans.labels_
logging.info("Clustering completed")

# Append cluster labels to the original dataframe
data['cluster'] = cluster_labels

# Save the dataframe with cluster labels to a new CSV file
data.to_csv("clustered_data.csv", index=False)
logging.info("Data with cluster labels saved to 'clustered_data.csv'")