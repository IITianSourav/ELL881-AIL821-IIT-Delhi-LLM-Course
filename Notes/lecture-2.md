

### **Fundamentals of NLP & Language Models**

1. **Definition and Scope of NLP**:
   - Natural Language Processing (NLP) bridges human language with machine learning, involving tasks from simple tokenization to complex dialogue management.

2. **Levels of NLP**:
   - **Phonetics/Phonology**: Focuses on sounds of speech and how they're interpreted.
   - **Morphology**: Deals with the structure of words (e.g., stems, roots, affixes).
   - **Syntax**: Studies sentence structure and grammatical rules.
   - **Semantics**: Focuses on the meaning of words and sentences.
   - **Pragmatics**: Considers context in interpreting language beyond literal meaning.
   - **Discourse**: Involves understanding language in larger context, such as paragraphs or dialogues.

3. **Language Models (LMs)**:
   - Predict the probability of a sequence of words. They are foundational for NLP tasks such as machine translation, speech recognition, and text generation.
   - Common approaches include **n-grams**, **Markov models**, and **neural networks**.

4. **Neural Language Models**:
   - Built using deep learning techniques (e.g., RNNs, LSTMs, transformers) for capturing long-term dependencies and context across large text corpora.
   - Transformer-based models, like BERT and GPT, have advanced NLP by enabling context-rich embeddings.

5. **Challenges in NLP**:
   - Ambiguity in language (words with multiple meanings).
   - Cultural and contextual nuances.
   - Processing large volumes of text data for training complex models.

---

### **Text Preprocessing & Basic Language Tasks**

1. **Preprocessing Steps**:
   - **Tokenization**: Breaking text into words or subwords.
   - **Stop Words Removal**: Removing common, insignificant words.
   - **Stemming/Lemmatization**: Reducing words to base or root forms.
   - **POS Tagging**: Labeling words with parts of speech (e.g., noun, verb).
   - **Chunking**: Identifying noun or verb phrases in sentences.

2. **Syntax in NLP**:
   - **Parsing**: Analyzing sentence structure (syntax trees) to understand grammatical relationships.
   - **Dependency Parsing**: Identifies dependencies between words, making relationships clearer (e.g., "He" is the subject of "likes").

3. **Semantics and Meaning Representation**:
   - Word Sense Disambiguation (WSD) differentiates meanings of ambiguous words based on context.
   - Named Entity Recognition (NER) identifies entities like names, locations, and dates.
   - Semantic Role Labeling (SRL) assigns roles (agent, patient, source) to words in sentences.

4. **Basic NLP Tasks**:
   - **Named Entity Recognition (NER)**: Identifying names, locations, etc., within text.
   - **Coreference Resolution**: Resolving references to the same entity (e.g., “he” refers to “John”).
   - **Information Extraction**: Extracting structured information from text, like events and relationships between entities.

---

### **WordNet, Distributional Semantics, and Advanced NLP Tasks**

1. **Ontological Semantics & WordNet**:
   - *Ontology*: Uses structured data to connect words based on relationships.
   - *WordNet*: A database where words are interlinked by relations (e.g., synonym, antonym, "is a," "part of").
   - **Challenge**: Building comprehensive ontologies is difficult due to language evolution and manual effort.

2. **Distributional Semantics**:
   - The meaning of a word is derived from its surrounding words in large corpora.
   - **Co-occurrence Matrix**: Measures how often words appear together in a defined window, providing context-based meaning.
   - **Vector Representation**: Each word has a vector, allowing similarity comparisons using cosine similarity or dot products.

3. **Pragmatics and Discourse in NLP**:
   - *Pragmatics*: The contextual meaning in communication (e.g., interpreting indirect requests).
   - *Discourse*: Deals with understanding context over larger text or conversations, crucial for effective dialogue systems.

4. **Advanced NLP Tasks**:
   - **Semantic Role Labeling (SRL)**: Assigns roles like agent, patient, etc., to words within sentences.
   - **Textual Entailment**: Determines if one sentence logically follows from another.
   - **Coreference Resolution**: Disambiguates references to entities (e.g., "he" referring to "John").
   - **Information Extraction**: Extracts structured information like names, dates, and events from unstructured text.
   - **Word Sense Disambiguation (WSD)**: Differentiates between meanings of ambiguous words.
   - **Sentiment Analysis**: Identifies the sentiment (positive, negative, neutral) in text.
   - **Machine Translation**: Translating text between languages, still a challenging problem due to bias and nuances.
   - **Summarization**:
      - *Extractive*: Summarizes by copying important sentences.
      - *Abstractive*: Generates new sentences that capture the essence of the original text.
      - *Aspect-Based*: Summarizes based on a specific focus (e.g., battery life in product reviews).
   - **Question Answering**: Answers factual, list-based, or descriptive questions from a text source.
   - **Dialogue Systems**: Manages ongoing conversations, handling context and topic shifts effectively.

---

These notes encapsulate the lecture's coverage of key NLP concepts, from foundational theories to specific tasks and challenges, providing a strong grounding in NLP's major areas and applications.
