# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:57:45 2023

@author: ElaheMsvi
"""

import tensorflow_datasets as tfds

ratings_dataset, ratings_dataset_info = tfds.load(
    name='movielens/100k-ratings',
    # MovieLens dataset is not splitted into `train` and `test` sets by default.
    # So TFDS has put it all into `train` split. We load it completely and split
    # it manually.
    split='train',
    # `with_info=True` makes the `load` function return a `tfds.core.DatasetInfo`
    # object containing dataset metadata like version, description, homepage,
    # citation, etc.
    with_info=True
)

import tensorflow as tf
assert isinstance(ratings_dataset, tf.data.Dataset)

print(
    "ratings_dataset size: %d" % ratings_dataset.__len__()
)

print(
    tfds.as_dataframe(ratings_dataset.take(5), ratings_dataset_info)
)

## Feature selection
ratings_dataset = ratings_dataset.map(
    lambda rating: {
        # `user_id` is useful as a user identifier.
        'user_id': rating['user_id'],
        # `movie_id` is useful as a movie identifier.
        'movie_id': rating['movie_id'],
        # `movie_title` is useful as a textual information about the movie.
        'movie_title': rating['movie_title'],
        # `user_rating` shows the user's level of interest to a movie.
        'user_rating': rating['user_rating'],
        # `timestamp` will allow us to model the effect of time.
        'timestamp': rating['timestamp']
    }
)

trainset_size = 0.8 * ratings_dataset.__len__().numpy()
# In an industrial recommender system, this would most likely be done by time:
# The data up to time T would be used to predict interactions after T.

# set the global seed:
tf.random.set_seed(42)
# More info: https://www.tensorflow.org/api_docs/python/tf/random/set_seed

# Shuffle the elements of the dataset randomly.
ratings_dataset_shuffled = ratings_dataset.shuffle(
    # the new dataset will be sampled from a buffer window of first `buffer_size`
    # elements of the dataset
    buffer_size=100_000,
    # set the random seed that will be used to create the distribution.
    seed=42,
    # `list(dataset.as_numpy_iterator()` yields different result for each call
    # Because reshuffle_each_iteration defaults to True.
    reshuffle_each_iteration=False
)
# More info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
 
ratings_trainset = ratings_dataset_shuffled.take(trainset_size)
ratings_testset = ratings_dataset_shuffled.skip(trainset_size)

print(
    "ratings_trainset size: %d" % ratings_trainset.__len__()
)
print(
    "ratings_testset size: %d" % ratings_testset.__len__()
)


# To make a custom `tf.data.Dataset` object including your own dataset visit [this link](https://www.tensorflow.org/datasets/add_dataset).
# 
# For more information about `tf.data.Dataset` visit [this link](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).


# Make a Keras Normalization layer to standardize a numerical feature.
timestamp_normalization_layer = \
    tf.keras.layers.experimental.preprocessing.Normalization(axis=None)

# Normalization layer is a non-trainable layer and its state (mean and std of
# feature set) must be set before training in a step called "adaptation".
timestamp_normalization_layer.adapt(
    ratings_trainset.map(
        lambda x: x['timestamp']
    )
)

for rating in ratings_trainset.take(3).as_numpy_iterator():
  print(
      f"Raw timestamp: {rating['timestamp']} ->",
      f"Normalized timestamp: {timestamp_normalization_layer(rating['timestamp'])}"
  )

# ### Turning categorical features into embeddings
# 
# A categorical feature is a feature that does not express a continuous quantity, but rather takes on one of a set of fixed values. Most deep learning models express these feature by turning them into high-dimensional embedding vectors which will be adjusted during model training.
# 
# Here we represent each user and each movie by an embedding vector. Initially, these embeddings will take on random values, but during training, we will adjust them so that embeddings of users and the movies they watch end up closer together.
# 
# Taking raw categorical features and turning them into embeddings is normally a two-step process:
# 
# 
# 1.   Build a mapping (called a `"vocabulary"`) that maps each raw values for example "Postman, The (1997)" to unique integers (say, 15).
# 2.   Turn these integers into embedding vectors.

user_id_lookup_layer = tf.keras.layers.StringLookup(mask_token=None)
user_id_lookup_layer.adapt(
    ratings_trainset.map(
    lambda x: x['user_id']
))

print(user_id_lookup_layer.get_vocabulary()[:10])


user_id_embedding_dim = 32
user_id_lookup_dim = user_id_lookup_layer.vocabulary_size() 

user_id_embeding_layer = tf.keras.layers.Embedding(
    input_dim =user_id_lookup_dim,
    output_dim=user_id_embedding_dim)

user_id_model = tf.keras.Sequential(
    [
     user_id_lookup_layer,
     user_id_embeding_layer
     ]
    )

print(
    "Embeddings for user ids: ['-2', '13', '655', 'xxx']\n",
    user_id_model(
        ['-2', '13', '655', 'xxx']
    )
)

movie_id_lookup_layer = tf.keras.layers.StringLookup(mask_token=None)
movie_id_lookup_layer.adapt(ratings_trainset.map(lambda x: x['movie_id']))

movie_id_lookup_dim = movie_id_lookup_layer.vocabulary_size()
movie_id_embedding_dim = 32

movie_id_embedding_layer = tf.keras.layers.Embedding(input_dim =movie_id_lookup_dim , 
                                                     output_dim =movie_id_embedding_dim )

movie_id_model = tf.keras.Sequential(
    [
     movie_id_lookup_layer,
     movie_id_embedding_layer
     ]
    )


movie_title_vectorization_layer = tf.keras.layers.TextVectorization()
movie_title_vectorization_layer.adapt(ratings_trainset.map(lambda x:x['movie_title']))

print(movie_title_vectorization_layer.get_vocabulary()[40:50])

movie_title_vec_size= movie_title_vectorization_layer.vocabulary_size()
movie_title_embeding_size = 16
movie_title_embeding_layer = tf.keras.layers.Embedding(input_dim = movie_title_vec_size,
                                                       output_dim = movie_title_embeding_size)

movie_title_model = tf.keras.Sequential(
    [
     movie_title_vectorization_layer,
     movie_title_embeding_layer,
     # each title contains multiple words, so we will get multiple embeddings
    # for each title that should be compressed into a single embedding for
    # the text. Models like RNNs, Transformers or Attentions are useful here.
    # However, averaging all the words' embeddings together is also a good
    # starting point.
    tf.keras.layers.GlobalAveragePooling1D()

     ]
    )

# Here we only used query and candidate identifiers to buid the towers. This
# corresponds exactly to a classic matrix factorization approach.
# https://ieeexplore.ieee.org/abstract/document/4781121
# Query tower
query_model = user_id_model 
# Candidate tower
candidate_model = movie_id_model

import tensorflow_recommenders as tfrs

movies_dataset, movies_data_info = tfds.load(
    name = 'movielens/100k-movies',
    with_info = True,
    split = 'train'
    )

# display(tfds.as_dataframe(movies_dataset.take(5) , movies_data_info))

candidates_corpus_dataset = movies_dataset.map(lambda x:x['movie_id'])

#-----------Ranking Task-----------------------------------------------------------------------------    
    
class RankingModel(tfrs.models.Model):
    def __init__(self, query_model,candidate_model):
        super().__init__()
        self.query_model: tf.keras.Model  =query_model
        self.candidate_model: tf.keras.Model = candidate_model
        self.rating_model: tf.keras.Model = tf.keras.Sequential(
            [tf.keras.layers.Dense(256, activation = 'relu'),
             tf.keras.layers.Dense(64 , activation = 'relu'),
             tf.keras.layers.Dense(1),
             ]
            )
        self.rating_task_layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError()
        ]
            )

    def compute_loss(self, features,training=False):
        query_embedding = self.query_model(features['user_id'])
        candidate_embeding = self.candidate_model(features['movie_id'])
        rating_prediction = self.rating_model(
            tf.concat([query_embedding,candidate_embeding],
            axis = 1))
        
        loss = self.rating_task_layer(
            predictions = rating_prediction, 
            labels = features['user_rating']
            )
        return loss
    

movielens_ranking_model = RankingModel(query_model,candidate_model) 

optimizer_step_size = 0.1
movielens_ranking_model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=optimizer_step_size)
    )   

ranking_ratings_trainset = ratings_trainset.shuffle(100_000).batch(8192).cache()
ranking_ratings_testset = ratings_testset.batch(4096).cache()

history = movielens_ranking_model.fit(
    ranking_ratings_trainset,
    validation_data=ranking_ratings_testset,
    validation_freq=1,
    epochs=5
)

import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model losses during training")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "test"], loc="upper right")
plt.show()
