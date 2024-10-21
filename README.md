
# Movie Recommendation System using ALS

## Overview

In this project, we use the Alternating Least Squares (ALS) algorithm with Spark APIs to predict movie ratings. The model is trained using the [ml-latest-small dataset](https://grouplens.org/datasets/movielens/latest/), which consists of user ratings for various movies.

The primary focus of this project is on collaborative filtering and employing the ALS algorithm to make movie recommendations based on user input.

## Alternating Least Squares (ALS)

ALS is a matrix factorization technique used in collaborative filtering, where a user-item matrix is decomposed into two lower-dimensional matrices (user and item matrices). The ALS algorithm learns latent factors that predict missing entries in the user-item matrix by performing matrix factorization. This technique is particularly effective for providing personalized recommendations.

The dataset used for training is contained in the following files:

- `links.csv`
- `movies.csv`
- `ratings.csv`
- `tags.csv`

## Prerequisites

- Python 3.8 or higher
- Apache Spark
- Jupyter Notebook (optional for development)
- PySpark

## Content

The project covers the following sections:
1. Load Data
2. Spark ALS-based approach for training the model
3. ALS Model Selection and Evaluation
4. Model Testing
5. Making movie recommendations

## Project Steps

### 1. Load Data

We start by loading the movie dataset files into Spark DataFrames and perform basic data inspection. The dataset consists of information about movies, users, and their respective ratings.

### 2. Spark ALS-based Approach for Training Model

In this step, we use the ALS algorithm to train the model. The data is split into training, validation, and test sets with a 60/20/20 ratio.

### 3. ALS Model Selection and Evaluation

We perform grid search to find the best hyperparameters (number of iterations, rank, and regularization) based on the Root Mean Squared Error (RMSE) on the validation set. The optimal hyperparameters found for this project are:
- Iterations: 10
- Rank: 12
- Regularization: 0.2

### 4. Model Testing

After model selection, we use the best model to make predictions on the test set and evaluate the performance of the model. The out-of-sample RMSE achieved is approximately 0.9006.

### 5. Make Movie Recommendations

A recommendation function is created to provide movie suggestions to a user based on their favorite movies. The function allows for input of a list of favorite movies and outputs a list of top 10 recommended movies.

## Code Example

Below is an example code snippet to make movie recommendations:

```python
# Define a list of favorite movies
my_favorite_movies = ['Iron Man', 'Toy Story']

# Get movie recommendations
recommends = make_recommendation(
    best_model_params={'iterations': 10, 'rank': 12, 'lambda_': 0.2}, 
    ratings_data=rating_data, 
    df_movies=movies, 
    fav_movie_list=my_favorite_movies, 
    n_recommendations=10, 
    spark_context=sc
)

# Print recommended movies
print(f"Recommendations for {my_favorite_movies}:")
for i, title in enumerate(recommends):
    print(f"{i + 1}: {title}")
```

## Dependencies

- PySpark
- Pandas
- NumPy
- Matplotlib

To install the required dependencies, use the following command:

```bash
pip install pyspark pandas numpy matplotlib
```

## Dataset

The dataset used in this project is the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/), which contains 100,000 ratings from 610 users on 9,742 movies.

## Running the Project

0. Go to [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/) and download the dataset.
1. Clone the repository:
   ```bash
   git clone https://github.com/Yoppman/movie_recommender_system.git
   cd movie-recommendation-system
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook or Python script to train the model and make recommendations.
