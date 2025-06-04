# Slime Mold Recommender System

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://cuilongyin.github.io/slime-mold-recommender/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org/)

A revolutionary recommendation system that combines **Variational Autoencoders (VAE)** with **bio-inspired slime mold dynamics** to deliver highly accurate personalized recommendations. Inspired by the intelligent foraging behavior of *Physarum polycephalum*, this algorithm discovers optimal pathways through user preference networks.

## Key Features

- **Bio-Inspired Algorithm**: Mimics slime mold foraging patterns for natural recommendation discovery
- **VAE Integration**: Advanced dimensionality reduction for scalable similarity computation  
- **High Performance**: Optimized with parallel processing and sparse matrix operations
- **Proven Results**: Achieves **0.867 RMSE** on MovieLens dataset with 100K+ ratings
- **Real-time**: Sub-second recommendation generation

## How It Works

### 1. VAE Compression
The system uses a Variational Autoencoder to compress high-dimensional user-item rating matrices into a compact 20-dimensional latent space, capturing essential preference patterns while handling sparsity effectively.

### 2. Similarity Network Construction  
Item similarities are computed in the latent space using cosine similarity, creating a weighted network where connections represent content relationships.

### 3. Slime Mold Simulation
Virtual slime mold agents explore the similarity network starting from user's liked items. The algorithm simulates:
- **Tube Growth**: Connections strengthen along frequently traveled paths
- **Decay**: Unused pathways weaken over time  
- **Reinforcement**: Popular routes receive more "flow"

### 4. Recommendation Generation
Items with the strongest tube connections to the user's preferences emerge as top recommendations, ensuring both relevance and diversity.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cuilongyin/slime-mold-recommender.git
cd slime-mold-recommender

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.slime_mold_recommender import SlimeMoldRecommenderV5
import pandas as pd

# Load your ratings data
ratings_df = pd.read_csv('your_ratings.csv')

# Create and configure the recommender
recommender = SlimeMoldRecommenderV5(
    decay_rate=0.1,      # Rate of tube decay
    growth_rate=0.2,     # Rate of tube growth
    n_iterations=10,     # Simulation iterations
    latent_dim=20,       # VAE latent dimensions
    vae_epochs=10,       # VAE training epochs
    debug=True
)

# Train the model
recommender.fit(ratings_df, 
               item_id_col='movieId', 
               user_id_col='userId', 
               rating_col='rating')

# Get recommendations for a user
liked_movies = [1, 42, 123, 456]  # Movie IDs the user liked
recommendations, scores = recommender.recommend(
    liked_items=liked_movies,
    top_n=5,
    excluded_items=[]  # Optional: exclude specific items
)

print("Recommended movies:", recommendations)
print("Recommendation scores:", scores)
```

### MovieLens Example

```python
# Run the complete MovieLens demonstration
python examples/movielens_demo.py
```

## Performance Results

Evaluated on the MovieLens dataset:

| Metric | Value |
|--------|-------|
| **RMSE** | 0.8670 |
| **Movies** | 9,724 |
| **Users** | 610 |
| **Ratings** | 100,836 |
| **Training Time** | ~5 minutes |
| **Recommendation Time** | <1 second |

## 🎛Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `decay_rate` | Rate at which unused tubes decay | 0.1 | 0.0-1.0 |
| `growth_rate` | Rate of tube reinforcement | 0.2 | 0.0-1.0 |
| `n_iterations` | Slime mold simulation steps | 10 | 5-50 |
| `latent_dim` | VAE latent space dimensions | 20 | 10-100 |
| `max_liked_items` | Max user history to consider | 50 | 10-200 |
| `max_neighbors` | Max similar items per item | 30 | 10-100 |

## Experiments & Evaluation

The system includes comprehensive evaluation capabilities:

```python
from sklearn.model_selection import train_test_split

# Split data for evaluation
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Evaluate the model
rmse = recommender.evaluate(
    train_data=train_data,
    test_data=test_data,
    k=10,           # Top-k recommendations
    n_workers=4     # Parallel processing
)

print(f"RMSE: {rmse:.4f}")
```

## Algorithm Comparison

| Algorithm | RMSE | Training Time | Recommendation Time |
|-----------|------|---------------|-------------------|
| **Slime Mold + VAE** | **0.867** | 5 min | <1 sec |
| Collaborative Filtering | 0.923 | 2 min | <1 sec |
| Matrix Factorization | 0.891 | 8 min | <1 sec |
| Neural Collaborative | 0.845 | 15 min | 2 sec |

## Project Structure

```
slime-mold-recommender/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   └── slime_mold_recommender.py     # Main algorithm implementation
├── examples/
│   ├── movielens_demo.py             # Complete demonstration
│   └── sample_data/                  # Sample datasets
├── docs/                             # GitHub Pages documentation
│   ├── index.html                    # Main documentation page
│   ├── demo.html                     # Interactive demo
│   └── api.html                      # API reference
├── results/
│   └── evaluation_results.md         # Performance benchmarks
└── tests/
    └── test_recommender.py           # Unit tests
```

## Interactive Demo

Try the algorithm yourself with our [interactive web demo](https://cuilongyin.github.io/slime-mold-recommender/demo.html)!

Features:
- Pre-loaded MovieLens dataset
- Real-time parameter tuning
- Visualization of slime mold network growth
- Performance metrics dashboard

## Scientific Background

This algorithm is inspired by several key research areas:

### Slime Mold Intelligence
- Based on *Physarum polycephalum* foraging behavior
- Implements tube network optimization principles
- Adapts biological reinforcement learning mechanisms

### Variational Autoencoders
- Learned latent representations of user preferences
- Handles sparse rating matrices effectively
- Enables scalable similarity computation

### Collaborative Filtering
- User-item interaction modeling
- Neighborhood-based recommendations
- Implicit feedback processing

## References & Inspiration

1. Tero, A., et al. (2010). "Rules for biologically inspired adaptive network design." *Science*, 327(5964), 439-442.
2. Kingma, D. P., & Welling, M. (2013). "Auto-encoding variational bayes." *arXiv preprint arXiv:1312.6114*.
3. Sarwar, B., et al. (2001). "Item-based collaborative filtering recommendation algorithms." *WWW '01*.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/cuilongyin/slime-mold-recommender.git
cd slime-mold-recommender

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run the example
python examples/movielens_demo.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MovieLens dataset provided by [GroupLens Research](https://grouplens.org/datasets/movielens/)
- Inspired by the fascinating research on slime mold intelligence
- Built with the amazing Python scientific computing ecosystem

## Contact

- **GitHub**: [@cuilongyin](https://github.com/cuilongyin)
- **Project Link**: [https://github.com/cuilongyin/slime-mold-recommender](https://github.com/cuilongyin/slime-mold-recommender)

---

**Star this repository** if you found it helpful!

**Found a bug?** [Open an issue](https://github.com/cuilongyin/slime-mold-recommender/issues)

**Have an idea?** [Start a discussion](https://github.com/cuilongyin/slime-mold-recommender/discussions)
