from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data (labeled and unlabeled)
labeled_data = [("This is a relevant web page", 1),
                ("This is not a relevant web page", 0),
                ("This is a positive review", 1),
                ("I really enjoyed this movie", 1),
                ("The food at this restaurant was terrible", 0),
                ("I would not recommend this product", 0),
                ("This is a positive sentence", 1),
                ("I really like this movie", 1),
                ("The food was terrible", 0),
                ("The service was great", 1),
                ("I had a terrible experience", 0),]

unlabeled_data = ["This is another web page",
                  "This is a third web page",
                  "The plot of this movie was confusing ",     "The service at this hotel was excellent ", "I didn't like the music in this concert", "I am not sure what to think about this product",
                  "The hotel was nice but the staff was rude", "The concert was amazing", "The traffic was terrible this morning", "I am not sure what to think about this product", "The hotel was nice but the staff was rude"]

# Split the labeled data into features and labels
labeled_features, labeled_labels = zip(*labeled_data)

# Convert the labeled data to a feature matrix using TF-IDF
vectorizer = TfidfVectorizer()
labeled_features_matrix = vectorizer.fit_transform(labeled_features)

# Train the semi-supervised model using LabelPropagation
semi_supervised_model = LabelPropagation(
    kernel='knn', n_neighbors=2, max_iter=1000)
semi_supervised_model.fit(labeled_features_matrix.toarray(), labeled_labels)

# Use the model to predict labels for the unlabeled data
unlabeled_features_matrix = vectorizer.transform(unlabeled_data)
predicted_labels = semi_supervised_model.predict(
    unlabeled_features_matrix.toarray())

# Print the predicted labels for the unlabeled data
print("Predicted labels for unlabeled data:", predicted_labels)

# Use the model to rank the web pages based on relevance to a query

query = "food"

all_data = labeled_data + [(page, None) for page in unlabeled_data]
all_features, _ = zip(*all_data)
all_features_matrix = vectorizer.transform(all_features)
rankings = semi_supervised_model.predict_proba(
    all_features_matrix.toarray())[:, 1]
ranked_data = [(all_data[i][0], rankings[i]) for i in range(len(all_data))]
sorted_data = sorted(ranked_data, key=lambda x: x[1], reverse=True)
relevant_pages = [page[0] for page in sorted_data if query in page[0].lower()]

# Print the relevant pages
print("Relevant pages:", relevant_pages)
