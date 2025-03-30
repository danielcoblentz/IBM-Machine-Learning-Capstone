# IBM-Machine-Learning-Capstone

A web-based course recommendation system that provides personalized course suggestions based on user preferences and similarity metrics. Built using [Streamlit](https://streamlit.io), a Python framework for building interactive web applications.


## Project Overview
This Course recommendation System that suggests relevant courses based on user ratings, course content, and clustering techniques. It utilizes machine learning algorithms, including [K-Means clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/), [TF-IDF vectorization](https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d/), and similarity metrics.


![demo](/images/app_demo.png)
<p align="center">Figure 1: Application demo</p>

### Features
 - Course Similarity-based recommendation
 - User Profile-based recommendation
 - Clustering-based recommendation (K-Means)
 - Clustering with PCA(principle componant analysis)
 - Interactive data display with Streamlit-AgGrid
 - Real-time recommendation updates

## Tech Stack

- Frontend: Streamlit
- Backend: Python, Pandas, NumPy
- Machine Learning: [Scikit-Learn](https://scikit-learn.org/stable/), [SciPy](https://docs.scipy.org/doc/scipy/)
- Data Storage: CSV files (ratings, courses, similarity matrices)
- Deployment: [Heroku](https://www.heroku.com)


## Course Topics Covered

### This project covers a wide range of machine learning techniques and concepts, including:

- Exploratory analysis of various datasets (course_genre, course_bows, ratings).
- Calculating course similarity using Bag of Words (BoW) features.
- Content-based course recommendation using user profiles and course genres.
- Clustering-based recommendation using K-Means.
- Collaborative filtering using K-Nearest Neighbor (KNN).
- Collaborative filtering using Non-Negative Matrix Factorization (NMF).
- Course similarity using neural networks.
- Regression-based rating score prediction using embedding features.
- Classification-based rating mode prediction using embedding features.



## There are a few different ways you can view this project:

1) Check the app deployed on Render [here](https://ibm-machine-learning-capstone.onrender.com)! As a note there are 4 models that are avaible for doing recommendations I do plan on adding more more of the models from the notebooks when I have time :)

2) Second you can go into the notebooks folder and check the notebooks they contain all model definitions, tests and a summary of their contents. I do recommend to open these in google colab.

3) Or you can run it locally on your own system by folowing these steps below.

    3.1 Installation
    
    Required files (data files should not be manipulated):
    - **`backend.py`**: 
    - **`recommendation_app.py`**: 
    - **`requirements.txt`**: .

    Steps to run the program:
    3.2. Clone the repository:
    ```bash
    git clone https://github.com/danielcoblentz/IBM-Machine-Learning-Capstone
    
    ```
    3.3 Install `requirements.txt`
    ```
    pip install -r requirements.txt
    ```
    3.4 Compile and run the recommender application:
    ```
    streamlit run recommender.py
    ```




 Notice:
 All content was completed through [coursera IBM Machine learning capstone course](https://www.coursera.org/learn/machine-learning-capstone/home/module/2)
 * All colab notebooks, datasets and streamlit starter files are sourced from the template provided by IBM as the basis for the web app, and notebooks.

