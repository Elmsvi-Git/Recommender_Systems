# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:11:07 2023

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

factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(
    # dataset of candidate embeddings from which candidates should be retrieved
    candidates=candidates_corpus_dataset.batch(128).map(candidate_model)
    )

retrieval_task_layer = tfrs.tasks.Retrieval(metrics= factorized_top_k_metrics)

class RetrievalModel(tfrs.models.Model):

    def __init__(self,query_model,candidate_model, retrival_task_layer):
        super().__init__()
        self.query_model:tf.models.Models = query_model
        self.candidate_model:tf.models.Models = candidate_model
        self.retrieval_task_layer:tf.models.Models = retrieval_task_layer
    
    def compute_loss(self , features, training=False)->tf.Tensor:
        query_embedding  = self.query_model(features['user_id'])
        posetiv_candidate_embedding = self.candidate_model(features['movie_id'])
        
        loss = self.retrieval_task_layer(query_embedding,posetiv_candidate_embedding)
        
        return loss
    
movielens_retrieval_model = RetrievalModel(query_model, 
                                           candidate_model,
                                           retrieval_task_layer)
## ------------------------------------------------------------------------------------------
optimizer_step_size = 0.1
movielens_retrieval_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=optimizer_step_size))

retrieval_ratings_trainset = ratings_trainset.map(lambda x:
                                                 {'user_id': x['user_id'],
                                                  'movie_id':x['movie_id']})
    
retrieval_ratings_testset = ratings_testset.map(lambda x:
                                                 {'user_id': x['user_id'],
                                                  'movie_id':x['movie_id']})
    
retrieval_cached_ratings_trainset = \
  retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
retrieval_cached_ratings_testset = \
  retrieval_ratings_testset.batch(4096).cache()


num_epochs = 5 
history = movielens_retrieval_model.fit(
    retrieval_cached_ratings_trainset,
    validation_data=retrieval_cached_ratings_testset,
    validation_freq=1,
    epochs=num_epochs
)
        
        
    
# Plot changes in model loss during training
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model losses during training")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "test"], loc="upper right")
plt.show()


# Plot changes in model accuracy during training
plt.plot(history.history["factorized_top_k/top_100_categorical_accuracy"])
plt.plot(history.history["val_factorized_top_k/top_100_categorical_accuracy"])
plt.title("Model accuracies during training")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train", "test"], loc="upper right")
plt.show()

#----------------------------------------------------------------------------------    
#Now that we have a model, we would like to be able to make predictions. 
# We can use the tfrs.layers.factorized_top_k.BruteForce layer to do this.


brute_force_layer = tfrs.layers.factorized_top_k.BruteForce(
    movielens_retrieval_model.query_model)
#adapting:
brute_force_layer.index_from_dataset(tf.data.Dataset.zip(
      candidates_corpus_dataset.batch(100),
      candidates_corpus_dataset.batch(100).map(movielens_retrieval_model.candidate_model)))

user_id = '42'
afinity_scores, movie_ids = brute_force_layer(tf.constant([user_id]))

print(f"Recommendations for user {user_id} using BruteForce: {movie_ids[0, :5]}")

#---Another method instead of Brute force: 
streaming_layer = tfrs.layers.factorized_top_k.Streaming(movielens_retrieval_model.query_model)
streaming_layer.index_from_dataset(tf.data.Dataset.zip(
      candidates_corpus_dataset.batch(100),
      candidates_corpus_dataset.batch(100).map(movielens_retrieval_model.candidate_model)))

user_id = '42'
afinity_scores, movie_ids = streaming_layer(tf.constant([user_id]))

print(f"Recommendations for user {user_id} using streaming_layer: {movie_ids[0, :5]}")

