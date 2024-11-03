
### Lecture on Word Representation in NLP

#### Recap of Previous Lecture
- Discussed **language modeling** in detail, including:
  - **Statistical language models**
  - **Smoothing techniques** for probability distribution
  - **Evaluation metrics** like *perplexity* (a measure tied to entropy)

#### Introduction to Semantics in NLP
- Transitioning to **semantics**, the study of meaning in language.
- **Goal:** Represent words, sentences, and tokens effectively.
- **Importance of context:** Words often have multiple meanings, so a single dictionary definition isn't sufficient.

---

### Key Updates in AI: Google DeepMind’s Alpha Models
- **Alpha Geometry 2** by DeepMind achieved impressive results in the International Math Olympiad, solving 4/6 problems, akin to a *silver medal*.
  - This success showcases the growing capacity of AI in **mathematical reasoning**.
  - However, smaller models may not replicate this level of success.

---

### Traditional Approaches to Word Representation

1. **Dictionary Definitions**:
   - **Dictionary example:** Merriam-Webster defines meaning as the idea or concept a word represents.
   - Challenge: Words like "bank" and "apple" have multiple meanings depending on context.

2. **Manual Ontologies**:
   - Early NLP approaches involved human-annotated ontologies (e.g., WordNet) linking words based on **semantic relations** like synonymy and antonymy.
   - **WordNet** organizes words by sense, with each sense connected through synonym sets or "synsets."
   - **Challenges**:
     - **Manual labor-intensive**: Requires extensive human input for accurate definitions.
     - **Coverage limitations**: Lacks recent words and certain parts of speech, and can be subjective.

3. **One-Hot Encoding**:
   - Represents each word by a unique vector where one position is “1” and all others are “0”.
   - **Drawbacks**:
     - High **dimensionality** (scales with vocabulary size).
     - **Orthogonality** of vectors means no similarity between words like "motel" and "hotel."

---

### Distributional Semantics: Foundation of Modern Word Representation

1. **Distributional Hypothesis**:
   - Proposed by linguist J.R. Firth: "You shall know a word by the company it keeps."
   - Words appearing in similar contexts likely have similar meanings.
  
2. **Count-Based Approaches**:
   - **Term-Context Matrix**: Counts co-occurrences of words within a context window across documents.
   - **Similarity Measurement**: Words with similar contexts are likely semantically similar.
   - **Challenges**:
     - High **dimensionality** and **sparsity** of the matrix.
     - Presence of **stop words** with inflated counts.

3. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - **TF** (Term Frequency): Measures how often a word appears in a document, adjusted using a log scale to reduce the effect of common words.
   - **IDF** (Inverse Document Frequency): Diminishes the weight of words that appear in many documents.
   - **Objective**: Emphasize unique, contextually relevant words and normalize the influence of frequent but uninformative words.

---

**Inverse Document Frequency (IDF):**
1. **Definition of Document Frequency (DF):**
   - Document frequency measures the number of documents in which a specific term appears.
   - Unlike term frequency, DF does not count the total occurrences within a document but simply whether the term appears or not.

2. **Examples of DF:**
   - Words with low DF are unique to specific documents (e.g., "Romeo" appearing in one document among many).
   - Words with high DF are common across multiple documents (e.g., "action" appearing in many documents).

3. **Inverse Document Frequency (IDF):**
   - IDF is calculated as the inverse of DF, often modified as $IDF = \log(\frac{N}{DF})$, where $N$ is the total number of documents.
   - The IDF value penalizes common words (those appearing in most documents), thus prioritizing unique or representative terms for the document.
   - Example: Words like "Romeo" (rare) have a high IDF, while common words like "good" have a low IDF, close to zero.

4. **TF-IDF Calculation:**
   - The final TF-IDF score is obtained by multiplying TF with IDF, adjusting each term’s weight within a document by its specificity across documents.
   - TF-IDF effectively filters out frequent, less informative words while highlighting significant, unique terms.

**Limitations of Count-Based Approaches:**
1. **Sparsity and Dimensionality:**
   - High dimensionality and sparsity arise as vocabulary grows, leading to larger matrices with numerous zero entries.
   - Computation complexity increases as matrix size grows.

2. **Alternative Approach - SVD (Singular Value Decomposition):**
   - SVD can reduce dimensionality in TF-IDF matrices but has high computational cost (quadratic time complexity) and must be re-run with vocabulary updates.

3. **Lack of Word Order Context:**
   - Count-based methods do not capture word ordering or context, treating words independently within a document.

---

**Prediction-Based Approaches - Word2Vec:**
1. **Transition to Prediction-Based Models:**
   - Developed in 2013 by Mikolov et al., Word2Vec introduced dense vector embeddings for words, providing compact representations (typically 100-500 dimensions).

2. **Advantages of Dense Vectors:**
   - Reduces overfitting risks with fewer parameters.
   - Can capture semantic similarities between words, e.g., "car" and "automobile."

3. **Two Key Models in Word2Vec:**
   - **Continuous Bag of Words (CBOW):** Predicts a target word based on surrounding context words.
   - **Skip-Gram Model:** Predicts surrounding context words given a target word (more popular and efficient with large corpora).

4. **CBOW vs. Skip-Gram:**
   - **CBOW** focuses on predicting a central word using its context, while **Skip-Gram** predicts the context words based on a central target word.
   - Skip-Gram is commonly combined with **Negative Sampling** to optimize for frequent word pairs while reducing computation.

5. **Skip-Gram Model with Negative Sampling:**
   - **Context Window and Sampling:** A sliding window of a fixed size (e.g., 10) is used to collect positive context pairs for the target word.
   - **Negative Sampling:** Non-context words outside the window are sampled as negative examples, often in larger numbers to enhance model accuracy.
   - Negative samples provide contrast to positive pairs, helping the model learn effective word embeddings.

---

**Mathematics of Word2Vec (Skip-Gram Model):**
1. **Objective:** 
   - Given a target word and a context word, the model calculates the probability that the word pair is a positive (contextual) instance.
   
2. **Classifier and Probability Calculation:**
   - Uses a logistic regression model with a sigmoid function:
     $\[
     P(\text{positive pair}) = \sigma(C \cdot W)
     \]$
   - Here, $C$ is the vector for the context word, and $W$ is the vector for the target word. The dot product measures similarity.

3. **Handling Multiple Word Pairs:**
   - For each context window, compute probabilities for all context pairs.
   - Take the log of probabilities to aggregate across pairs, creating an efficient loss function for training.

4. **Optimization and Training:**
   - Iterative updates refine both target and context embeddings, yielding a model that captures semantic relationships across words in dense vectors.
   
---

### Summary and Conclusion

In this discussion, we explored the methodology of word embeddings, particularly through the lens of the Skip-Gram model and its enhancements. We initiated with a random initialization of the embedding matrix, characterized by a size of $2V \times D$, where $V$ denotes the vocabulary size and $D$ the dimension of the embeddings. 

We meticulously curated positive and negative instances using context windows. By applying logistic regression, we aimed to maximize the likelihood of positive word-context pairs while minimizing that of negative pairs. We incorporated a modified unigram probability to effectively sample negative instances, ensuring that less frequent words are given a fair chance during the sampling process. 

Furthermore, we implemented subsampling to reduce the impact of high-frequency, non-meaningful words, such as articles, which do not contribute significantly to the semantic understanding of text.

One of the notable advantages of our approach is the capability to perform word analogies. For instance, by leveraging vector arithmetic, we can represent relationships such as "king - man + woman = queen," demonstrating the semantic relationships captured by the embeddings.

However, we also recognized the limitations of static embeddings, such as their inability to adapt to different meanings of words in varying contexts and challenges with out-of-vocabulary words. To address these, we introduced subword embeddings, exemplified by the FastText model. This model utilizes character n-grams to create embeddings for words, enabling the system to derive embeddings for unseen words based on their subword components.

In conclusion, the evolution of word embedding techniques has greatly enhanced our ability to capture the nuances of language, paving the way for more sophisticated natural language processing applications. Future discussions will delve into alternative models, such as GloVe, to further expand on the topic of word embeddings.
