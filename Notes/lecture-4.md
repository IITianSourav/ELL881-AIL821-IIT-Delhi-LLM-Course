## Advanced Smoothing & Evaluation 


#### Advanced Smoothing Algorithms

-Naïve smoothing algorithms have limited usage and are not very effective. Not frequently used for N-grams.

-However, they can be used in domains where the number of zeros isn't so huge.

Popular Algorithems:
- Good-Turing
- Kneser-Ney

#### Good-Turing Algorithem 

  - **Intution**: We use the count of things we've seen once to help estimate the count of things we have never seen.

  - **Example**: You are birdwatching in the Jim Corbett National Park and you have observed the following birds: 10 Flamingos, 3 Kingfishers, 2 Indian Rollers, 1 Woodpecker, 1 Peacock, 1 Crane = 18 birds

     How likely is it that the next bird you see is a woodpecker?
     1/18
     How likely is it that the next bird you see is a new species -- Purple Heron or Painted Stork?

     We will use our estimate of things we saw once to estimate the new things.
     3/18 (because N₁ = 3)

Kneser-Ney smoothing is a popular technique used in natural language processing (NLP) for estimating probabilities of word sequences (n-grams) in language models. It is particularly effective for handling sparse data, which is common in NLP tasks. Below is a detailed explanation, including the formula, definition, intuition, and an explanation of the algorithm.

#### Kneser-Ney Smoothing

Kneser-Ney smoothing is a method used to improve the estimation of probabilities for n-grams, especially when dealing with unseen or rare sequences. It modifies the way probabilities are calculated for n-grams by taking into account the frequency of lower-order n-grams and their contextual information.

### Formula

The Kneser-Ney smoothing algorithm consists of two main components: **discounting** and **back-off**. The probabilities are computed as follows:

1. **Discounted Probability** for a seen n-gram:
   $$\[
   P_{KN}(w_n | w_1, w_2, \ldots, w_{n-1}) = \max\left( \frac{C(w_1, w_2, \ldots, w_n) - D}{C(w_1, w_2, \ldots, w_{n-1})}, 0 \right) + \lambda(w_1, w_2, \ldots, w_{n-1}) P_{KN}(w_n | w_2, \ldots, w_{n-1})
   \]$$

   Where:
   - $\( C(w_1, w_2, \ldots, w_n) \)$ is the count of the n-gram.
   - $\( D \)$ is the discounting factor (usually between 0.5 and 1).
   - $\( \lambda(w_1, w_2, \ldots, w_{n-1}) \)$is a normalization constant.
   - The second term is the back-off probability from the (n-1)-gram.

2. **For unseen n-grams**:
   $$\[
   P_{KN}(w_n | w_1, w_2, \ldots, w_{n-1}) = \lambda(w_1, w_2, \ldots, w_{n-1}) P_{KN}(w_n | w_2, \ldots, w_{n-1})
   \]$$

### Intuition

The intuition behind Kneser-Ney smoothing can be understood as follows:

1. **Discounting**: In Kneser-Ney, when calculating the probability of an n-gram, a fixed amount $\( D \)$ is subtracted from the count of each n-gram to adjust for overestimating probabilities due to finite data. This discounting allows for the redistribution of probability mass from more frequent n-grams to less frequent ones, thus handling the problem of zero probabilities for unseen n-grams.

2. **Back-off**: If the n-gram count after discounting becomes zero, the model can fall back on a lower-order n-gram probability. This ensures that even if a particular sequence of words has not been seen in the training data, we can still estimate its probability based on the frequency of shorter sequences.

3. **Contextual Consideration**: Kneser-Ney also incorporates the concept of lower-order n-grams by considering how many different contexts a word appears in, thereby improving the estimate for rare words or sequences. It effectively captures how often a word is used in various contexts rather than just in a fixed sequence.

---

##  Evaluation of Langauge Models

- In machine learning, particularly in natural language processing (NLP), the evaluation of language models is crucial for understanding their performance.
- A common challenge is deciding between different models (e.g., L1 and L2) without sophisticated frameworks or metrics.


#### Key Concepts in Language Models
1. **Intrinsic and Implicit Measurement**:
   - Language models must be assessed intrinsically or implicitly to gauge their quality, as direct comparisons may not always be feasible.
   - This helps in selecting the best model in the tranning itself.

---

2. **Perplexity as a Metric**:
   - **Definition**: Perplexity is a measurement used to evaluate language models based on their ability to predict a sequence of words. It can be understood through the analogy of a guessing game.
   - **Intuition**: The more context provided, the easier it is for the model to make predictions. If a model can accurately predict the next word in a sequence with high probability, it results in low perplexity.
   - **Formula**: For a test set of tokens $\( W_1, W_2, \ldots, W_n \)$:
     
     $$\text{PP}(W) = P(W_1, W_2, \ldots, W_n)^{-1/n}$$

     - This implies that a model producing a higher probability results in lower perplexity.

---

3. **Entropy and its Relationship with Perplexity**:
   - **Entropy**: Measures the average uncertainty in a random variable. For language, it can be defined as:
     $H(X) = -\sum P(X_i) \log(P(X_i))$
   - **Entropy Rate**: This is the average entropy per word in a sentence:
     $H(L) = \lim_{n \to \infty} \frac{1}{n} H(W_1, W_2, \ldots, W_n)$
   - **Cross Entropy**: Measures the difference between two probability distributions. For a language model $\( M \)$ approximating true language $\( L \)$:
     $H(L, M) = -\sum P_L(x_i) \log(P_M(x_i))$
   - The relationship between perplexity and entropy can be expressed as:
     $\text{PP} = 2^{H}$
   - This shows that perplexity reflects the number of possible words the model can generate, where a higher entropy indicates greater uncertainty and a larger set of potential next words.

---

#### Importance of Perplexity
- While entropy gives a measure of average bits required to encode information, perplexity provides a more intuitive metric. It tells us how many potential words a language model can predict at each step.
- Lower perplexity suggests a more effective language model, as it indicates better predictive power and reduced uncertainty.

---

#### Challenges with Statistical Language Models
- **Fixed Window Size**: Selecting an appropriate window size for n-gram models is not straightforward and can limit the model's effectiveness.
- **Sparse Data**: With increased n-grams, the matrix size grows, leading to many zero counts, making it difficult to handle unseen words.
- **Computational Cost**: Creating large n-gram tables is computationally expensive, especially with massive corpora.

---

#### Transition to More Advanced Models
- The lecture concluded with an introduction to the need for improved representations that capture context more effectively.
- **Next Steps**: Upcoming discussions will focus on word embedding methods, including:
  - Distributional semantics
  - Limitations of distributional semantics
  - Introduction to models like Word2Vec and GloVe
  - Exploration of neural language models

---

### Conclusion
- A comprehensive understanding of evaluation metrics, particularly perplexity, is essential for assessing language models in NLP.
- Moving forward, the focus will shift to more advanced methods that aim to overcome the limitations of traditional statistical models.

--- 
