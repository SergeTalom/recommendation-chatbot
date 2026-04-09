Smart Course Recommender Chatbot

Overview
This project is a conversational learning-resource recommender built with Streamlit. It helps users discover suitable learning materials by understanding a natural-language query and recommending resources that match the user's topic of interest, learner level, and preferred study format.

The system supports interactive recommendations through a chatbot interface. It can understand user intent, recommend a primary learning resource, ask for clarification when needed, and provide additional related recommendations.

Main Features
Chat-style recommendation interface built with Streamlit
Intent understanding using classification and rule-based extraction
Support for topic, learner level, and resource-type prediction
TF-IDF and cosine-similarity-based primary recommendation
Secondary recommendations based on cluster-aware follow-up logic
Clarification flow for vague or underspecified queries

Supported Resource Types
Book
Course
Research Paper
YouTube Video

Project Structure
app.py: Streamlit user interface and conversational flow
recommender_logic.py: intent understanding, classification, filtering, similarity ranking, and follow-up recommendation logic
resources.csv: learning-resource dataset used for training and recommendation
ClusteringXDatabase_Creation.ipynb: Clustering work and database (i.e resources.csv) creation 
requirements.txt: project dependencies

How the System Works

User Query: The user enters a natural-language request such as: I want a beginner book on regression.
Intent Understanding: The system analyzes the query and identifies the relevant topic, learner level, and resource type. This is done using a combination of direct keyword extraction and machine-learning-based text classification.

Three classification tasks are performed: topic classification, learner-level classification, and resource-type classification. The code trains multiple traditional machine learning models for each task: Logistic Regression, SGD Classifier, and Multinomial Naive Bayes. If the confidence of a prediction is low, the system asks the user for clarification instead of making a weak recommendation.
Candidate Filtering: After intent is identified, the dataset is filtered using the predicted topic and, when appropriate, learner level and resource type.
Similarity-Based Primary Recommendation: For recommendation, the title and description of each resource are combined into a single text field. Both the user query and candidate resources are converted into TF-IDF vectors. Cosine similarity is then calculated to measure how closely each candidate matches the query. The ranking is further refined by giving extra preference to resources that match the predicted learner level and resource type. The top-ranked result becomes the primary recommendation shown to the user.
Secondary Recommendation: After a primary recommendation is made, the chatbot can provide additional related resources. This is supported through cluster-based follow-up logic using the sent_clusters field in the dataset.

Dataset
The project uses a structured CSV file named resources.csv. Each record represents a learning resource and includes fields such as resource type, learner level, relevant topic, title, description, release date, creator name, and cluster label.

The dataset is used both as the recommendation database and as labeled training data for the classification subsystem.

Running the Project
1. Open the project folder
Open the extracted recommendation-chatbot folder in your code editor or terminal.

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py

The app will usually open at http://localhost:8501

Dependencies
	- Streamlit
	- pandas
	- scikit-learn

Example Interaction
User query: I want to learn about classification.
Possible system behavior: ask the user to clarify learner level and resource type, recommend a suitable primary resource after clarification, and offer more related recommendations if the user asks for more.

Current Limitations
	- Only a limited set of resource types is supported
	- Subjective preferences such as teaching style, price, ratings, or duration are not modeled
	- Follow-up conversation handling depends on recognized phrase patterns
	- Recommendation quality depends on the coverage and balance of the dataset

Future Improvements
	- Expand the dataset with richer metadata
	- Support additional resource types and user preferences
	- Improve follow-up intent handling for more natural conversations
	- Strengthen ranking and personalization logic



