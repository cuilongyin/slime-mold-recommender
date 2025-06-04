import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from tqdm import tqdm
import time

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


# Try importing tensorflow, with a fallback to sklearn for dimensionality reduction
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam
    USE_VAE = True
except ImportError:
    from sklearn.decomposition import TruncatedSVD
    USE_VAE = False


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        # Compute KL divergence loss here
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SlimeMoldRecommenderV5:
    """
    A recommender system using VAE-enhanced collaborative filtering with slime mold dynamics.
    Uses a VAE to compress the rating data before computing similarities.
    """

    def __init__(self, decay_rate=0.1, growth_rate=0.2, n_iterations=10,
                 max_liked_items=50, max_neighbors=30, latent_dim=20,
                 vae_epochs=10, debug=False):
        """
        Initialize the VAE-enhanced slime mold recommender.

        Parameters:
        -----------
        decay_rate : float
            Rate at which slime tubes decay when not used
        growth_rate : float
            Rate at which slime tubes grow when reinforced
        n_iterations : int
            Number of iterations for slime growth simulation
        max_liked_items : int
            Maximum number of liked items to consider per user
        max_neighbors : int
            Maximum number of neighbors to consider per item
        latent_dim : int
            Dimensionality of the VAE latent space
        vae_epochs : int
            Number of epochs to train the VAE
        debug : bool
            Whether to print debug information
        """
        self.decay_rate = decay_rate
        self.growth_rate = growth_rate
        self.n_iterations = n_iterations
        self.max_liked_items = max_liked_items
        self.max_neighbors = max_neighbors
        self.latent_dim = latent_dim
        self.vae_epochs = vae_epochs
        self.debug = debug

        # Item data
        self.n_items = 0
        self.n_users = 0
        self.item_ids = None
        self.user_ids = None
        self.id_to_idx = None
        self.user_id_to_idx = None

        # Compressed embeddings and similarity data
        self.item_embeddings = None
        self.nearest_neighbors = None
        self.vae = None

    def _build_vae(self, input_dim):
        """
        Build a Variational Autoencoder for compressing rating vectors.

        Parameters:
        -----------
        input_dim : int
            Dimensionality of input data (number of users)

        Returns:
        --------
        vae : Model
            The compiled VAE model
        encoder : Model
            The encoder part of the VAE
        """
        if not USE_VAE:
            return None, None

        # Encoder
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(min(input_dim, 128), activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(min(input_dim, 64), activation='relu')(x)

        # Latent space
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)

        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        #z = layers.Lambda(sampling)([z_mean, z_log_var])
        z = Sampling()([z_mean, z_log_var])
        # Encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(min(input_dim, 64), activation='relu')(latent_inputs)
        x = layers.Dense(min(input_dim, 128), activation='relu')(x)
        outputs = layers.Dense(input_dim, activation='sigmoid')(x)

        # Decoder model
        decoder = Model(latent_inputs, outputs, name="decoder")

        # VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name="vae")

        # Add KL divergence loss
        #kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        #vae.add_loss(kl_loss)

        # Compile VAE
        vae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return vae, encoder

    def fit(self, ratings_df, item_id_col='movieId', user_id_col='userId', rating_col='rating'):
        """
        Fit the recommender with ratings data.

        Parameters:
        -----------
        ratings_df : DataFrame
            Dataframe containing user ratings
        item_id_col : str
            Name of the column containing item IDs
        user_id_col : str
            Name of the column containing user IDs
        rating_col : str
            Name of the column containing ratings
        """
        start_time = time.time()
        print("Preparing data structures...")

        # Get unique items and users
        unique_items = ratings_df[item_id_col].unique()
        unique_users = ratings_df[user_id_col].unique()

        self.n_items = len(unique_items)
        self.n_users = len(unique_users)

        self.item_ids = unique_items
        self.user_ids = unique_users

        self.id_to_idx = {id: idx for idx, id in enumerate(unique_items)}
        self.user_id_to_idx = {id: idx for idx, id in enumerate(unique_users)}

        print(f"Dataset has {self.n_items} items and {self.n_users} users")

        # Create user-item rating matrix (sparse)
        print("Creating user-item matrix...")

        rows = []
        cols = []
        values = []

        for _, row in ratings_df.iterrows():
            user_idx = self.user_id_to_idx[row[user_id_col]]
            item_idx = self.id_to_idx[row[item_id_col]]
            rating = row[rating_col]

            rows.append(user_idx)
            cols.append(item_idx)
            values.append(rating)

        # Create sparse matrix
        rating_matrix = csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

        # For VAE, we want item x user matrix (transpose)
        item_user_matrix = rating_matrix.T.toarray()

        # Normalize ratings to [0, 1] for VAE
        rating_max = ratings_df[rating_col].max()
        rating_min = ratings_df[rating_col].min()
        item_user_matrix_norm = (item_user_matrix - rating_min) / (rating_max - rating_min)

        # Handle missing ratings (NaN values)
        item_user_matrix_norm = np.nan_to_num(item_user_matrix_norm)

        print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")

        # Create and train VAE or use SVD as fallback
        print("Compressing rating data...")
        start_time = time.time()

        if USE_VAE:
            print("Training VAE...")
            self.vae, encoder = self._build_vae(self.n_users)

            # Train VAE
            self.vae.fit(
                item_user_matrix_norm,
                item_user_matrix_norm,
                epochs=self.vae_epochs,
                batch_size=256,
                shuffle=True,
                verbose=1
            )

            # Get item embeddings from encoder
            _, _, self.item_embeddings = encoder.predict(item_user_matrix_norm)

        else:
            print("TensorFlow not available. Using SVD for dimensionality reduction...")
            # Use SVD as fallback
            svd = TruncatedSVD(n_components=self.latent_dim)
            self.item_embeddings = svd.fit_transform(item_user_matrix)

        # Calculate the full similarity matrix at once
        similarity_matrix = cosine_similarity(self.item_embeddings)
        similarity_matrix = (similarity_matrix + 1) / 2  # Convert cosine values from [-1,1] to [0,1]

        # Remove self-similarity
        np.fill_diagonal(similarity_matrix, 0)

        # Build the nearest neighbors list for each item using a threshold of 0.1
        self.nearest_neighbors = []
        for i in range(self.n_items):
            # Collect neighbors with similarity above threshold
            neighbors = [(j, similarity_matrix[i, j]) for j in range(self.n_items) if similarity_matrix[i, j] > 0.1]
            # Sort neighbors in descending order and keep top ones
            neighbors.sort(key=lambda x: x[1], reverse=True)
            self.nearest_neighbors.append(neighbors[:self.max_neighbors])

        print(f"Compression completed in {time.time() - start_time:.2f} seconds")

        # Calculate item similarities based on embeddings
        print("Calculating item similarities based on embeddings...")
        start_time = time.time()


        print(f"Similarity calculation completed in {time.time() - start_time:.2f} seconds")
        return self

    def _simulate_slime_growth(self, liked_items):
        """
        Slime mold growth simulation on the embedding-based similarity network.

        Parameters:
        -----------
        liked_items : list
            Indices of items the user has liked

        Returns:
        --------
        dict : {item_idx: score}
            Dictionary mapping item indices to recommendation scores
        """
        # Limit the number of liked items for performance
        if len(liked_items) > self.max_liked_items:
            liked_items = np.random.choice(liked_items, self.max_liked_items, replace=False)

        # Create mini-graph with relevant items
        relevant_items = set(liked_items)

        # Expand network with neighbors of liked items
        for item_idx in liked_items:
            if item_idx < len(self.nearest_neighbors):
                for neighbor_idx, _ in self.nearest_neighbors[item_idx]:
                    relevant_items.add(neighbor_idx)

        # Convert to list for indexing
        relevant_items = list(relevant_items)
        n_relevant = len(relevant_items)

        # Create mapping between original indices and positions in subgraph
        idx_to_pos = {idx: pos for pos, idx in enumerate(relevant_items)}

        # Initialize flow (1.0 for liked items, 0 for others)
        flow = np.zeros(n_relevant)
        for item_idx in liked_items:
            if item_idx in idx_to_pos:
                flow[idx_to_pos[item_idx]] = 1.0

        # Create sparse tubes matrix using dictionaries
        tubes = {}
        for i, orig_i in enumerate(relevant_items):
            if orig_i >= len(self.nearest_neighbors):
                continue  # Skip if no neighbors data

            tubes[i] = {}

            # Add connections to neighbors if they're in our subgraph
            for neighbor_idx, sim in self.nearest_neighbors[orig_i]:
                if neighbor_idx in idx_to_pos:
                    j = idx_to_pos[neighbor_idx]
                    tubes[i][j] = sim

        # Simulate slime mold growth
        for _ in range(self.n_iterations):
            # Calculate pressure at each node
            pressure = np.zeros(n_relevant)

            # Distribute flow from each node to its neighbors
            for i in range(n_relevant):
                if flow[i] > 0.01 and i in tubes:  # Only for nodes with flow
                    total_strength = sum(tubes[i].values()) or 1.0
                    for j, strength in tubes[i].items():
                        pressure[j] += flow[i] * strength / total_strength

            # Update flow - liked items always have flow=1.0
            new_flow = pressure.copy()
            for item_idx in liked_items:
                if item_idx in idx_to_pos:
                    new_flow[idx_to_pos[item_idx]] = 1.0

            # Apply decay and growth to tubes
            for i in tubes:
                for j in tubes[i]:
                    # Decay
                    tubes[i][j] *= (1 - self.decay_rate)

                    # Growth (if both nodes have significant flow)
                    if flow[i] > 0.1 and flow[j] > 0.1:
                        tubes[i][j] += self.growth_rate * flow[i] * flow[j]

            # Update flow for next iteration
            flow = new_flow

        # Calculate final scores
        scores = {}

        # For each relevant item, calculate its recommendation score
        for pos, item_idx in enumerate(relevant_items):
            # Skip liked items
            if item_idx in liked_items:
                continue

            # Calculate score based on connections from liked items
            score = 0
            for liked_item in liked_items:
                if liked_item in idx_to_pos:
                    liked_pos = idx_to_pos[liked_item]

                    # Add direct connection strength if exists
                    if liked_pos in tubes and pos in tubes[liked_pos]:
                        score += tubes[liked_pos][pos] * flow[liked_pos]

            # Only include positive scores
            if score > 0:
                scores[item_idx] = score

        return scores

    def recommend(self, liked_items, top_n=5, excluded_items=None):
        """
        Get recommendations based on VAE embeddings and slime mold simulation.

        Parameters:
        -----------
        liked_items : list
            IDs of items the user has liked
        top_n : int
            Number of recommendations to return
        excluded_items : list, optional
            IDs of items to exclude from recommendations

        Returns:
        --------
        recommendations : list
            IDs of recommended items
        scores : list
            Recommendation scores
        """
        start_time = time.time()

        # Convert item IDs to indices
        liked_indices = [self.id_to_idx.get(item_id, -1) for item_id in liked_items]
        liked_indices = [idx for idx in liked_indices if idx >= 0]

        if not liked_indices:
            return [], []  # No valid liked items

        # Set up excluded items
        if excluded_items is None:
            excluded_items = []

        excluded_indices = [self.id_to_idx.get(item_id, -1) for item_id in excluded_items]
        excluded_indices = [idx for idx in excluded_indices if idx >= 0]

        # Add liked items to excluded items
        all_excluded = set(liked_indices + excluded_indices)

        if self.debug:
            print(f"Starting slime growth simulation with {len(liked_indices)} liked items")

        # Simulate slime growth and get scores
        score_dict = self._simulate_slime_growth(liked_indices)

        # Remove excluded items
        for idx in all_excluded:
            if idx in score_dict:
                del score_dict[idx]

        if self.debug:
            print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

        # Sort by score and get top N
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if not sorted_items:
            return [], []  # No recommendations found

        # Split into recommendations and scores
        rec_indices, scores = zip(*sorted_items) if sorted_items else ([], [])

        # Convert indices back to item IDs
        recommendations = [self.item_ids[idx] for idx in rec_indices]

        if self.debug:
            print(f"Recommendation generation completed in {time.time() - start_time:.2f} seconds")

        return recommendations, list(scores)

    def _process_user_batch(self, user_batch, train_data, test_data, k,
                           user_id_col='userId', item_id_col='movieId', rating_col='rating'):
        """
        Process a batch of users for evaluation.

        Returns:
        --------
        tuple : (predictions, actuals)
            Lists of predicted and actual ratings
        """
        batch_predictions = []
        batch_actuals = []

        for user_id in user_batch:
            # Get test items for this user
            user_test = test_data[test_data[user_id_col] == user_id]
            if len(user_test) == 0:
                continue

            # Get training items for this user
            user_train = train_data[train_data[user_id_col] == user_id]
            if len(user_train) == 0:
                continue

            # Get items user liked in training (rating >= 4)
            liked_items = user_train[user_train[rating_col] >= 4][item_id_col].tolist()
            if len(liked_items) == 0:
                continue

            # Items to exclude (all rated by user in training)
            excluded_items = user_train[item_id_col].tolist()

            try:
                # Get recommendations
                recs, scores = self.recommend(
                    liked_items=liked_items,
                    top_n=k,
                    excluded_items=excluded_items
                )

                # Normalize scores to 1-5 range for prediction
                if scores:
                    max_score = max(scores)
                    if max_score > 0:
                        scaled_scores = [1 + 4 * (score / max_score) for score in scores]
                    else:
                        scaled_scores = [3.0] * len(scores)
                else:
                    scaled_scores = []

                # Match recommendations with test items
                for rec, score in zip(recs, scaled_scores):
                    test_item = user_test[user_test[item_id_col] == rec]
                    if len(test_item) > 0:
                        batch_predictions.append(score)
                        batch_actuals.append(test_item[rating_col].values[0])

            except Exception as e:
                if self.debug:
                    print(f"Error processing user {user_id}: {e}")
                continue

        return batch_predictions, batch_actuals

    def evaluate(self, train_data, test_data, k=5, n_workers=4,
                user_id_col='userId', item_id_col='movieId', rating_col='rating'):
        """
        Evaluate the recommender using RMSE with parallel processing.

        Parameters:
        -----------
        train_data : DataFrame
            Training data with user ratings
        test_data : DataFrame
            Test data with user ratings
        k : int
            Number of recommendations to make per user
        n_workers : int
            Number of parallel workers for evaluation

        Returns:
        --------
        rmse : float
            Root Mean Squared Error
        """
        # Get unique users from test data
        test_users = test_data[user_id_col].unique()

        # Create batches of users
        batch_size = 5
        user_batches = [test_users[i:i + batch_size] for i in range(0, len(test_users), batch_size)]

        predictions = []
        actuals = []

        # Process batches with progress bar
        with tqdm(total=len(user_batches), desc="Evaluating user batches") as pbar:
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_results = [
                        executor.submit(
                            self._process_user_batch,
                            batch, train_data, test_data, k,
                            user_id_col, item_id_col, rating_col
                        ) for batch in user_batches
                    ]

                    for future in future_results:
                        try:
                            batch_predictions, batch_actuals = future.result()
                            predictions.extend(batch_predictions)
                            actuals.extend(batch_actuals)
                            pbar.update(1)
                        except Exception as e:
                            if self.debug:
                                print(f"Error in batch processing: {e}")
                            pbar.update(1)
            else:
                # Sequential processing
                for batch in user_batches:
                    batch_predictions, batch_actuals = self._process_user_batch(
                        batch, train_data, test_data, k,
                        user_id_col, item_id_col, rating_col
                    )
                    predictions.extend(batch_predictions)
                    actuals.extend(batch_actuals)
                    pbar.update(1)

        # Calculate RMSE
        if predictions:
            rmse = sqrt(mean_squared_error(actuals, predictions))
            return rmse
        else:
            return None


# Example usage with MovieLens data
if __name__ == "__main__":
    try:
        print("Loading MovieLens dataset...")
        # Load data
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')

        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")

        # Create recommender
        print("Fitting recommender...")
        recommender = SlimeMoldRecommenderV5(
            decay_rate=0.1,
            growth_rate=0.2,
            n_iterations=10,
            max_liked_items=50,
            max_neighbors=30,
            latent_dim=20,   # Dimension of VAE latent space
            vae_epochs=10,   # VAE training epochs
            debug=True
        )

        # Fit with ratings data
        recommender.fit(ratings_df)

        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        train_ratings, test_ratings = train_test_split(
            ratings_df, test_size=0.2, random_state=42
        )

        # Evaluate the model
        print("Evaluating recommender...")
        rmse = recommender.evaluate(
            train_ratings,
            test_ratings,
            k=10,
            n_workers=4
        )

        if rmse:
            print(f"RMSE: {rmse:.4f}")
        else:
            print("Evaluation failed - no matching predictions")

        # Get recommendations for a specific user
        user_id = 1
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        liked_movie_ids = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()

        if len(liked_movie_ids) > 5:
            # If user likes many movies, just use a subset for clarity
            liked_movie_ids = liked_movie_ids[:5]

        # Generate recommendations
        print(f"\nGenerating recommendations for user {user_id}...")
        recommendations, scores = recommender.recommend(
            liked_movie_ids,
            top_n=5,
            excluded_items=user_ratings['movieId'].tolist()
        )

        # Print liked movies
        print(f"User {user_id} liked movies:")
        for movie_id in liked_movie_ids:
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            print(f"  - {movie['title']} ({movie['genres']})")

        # Print recommended movies
        print("\nRecommended movies:")
        for movie_id, score in zip(recommendations, scores):
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            print(f"  - {movie['title']} ({movie['genres']}) - Score: {score:.2f}")

    except FileNotFoundError:
        print("MovieLens dataset not found. Please download from: https://grouplens.org/datasets/movielens/")
