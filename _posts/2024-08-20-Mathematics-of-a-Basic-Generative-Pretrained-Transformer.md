---
usemathjax: true
title: "Mathematical Foundations of Building a Basic Generative Pretrained Transformer"
excerpt: "Building a Basic GPT Model from Scratch in Python: A Mathematical Breakdown with a Practical Example"

header:
  image: "./../assets/images/posts/2024-08-20-Mathematics-of-a-Basic-Generative-Pretrained-Transformer/logo3.jpg"
  teaser: "./../assets/images/posts/2024-08-20-How-to-Build-a-basic-LLM-GPT-from-Scratch-in-Python/gpt-logo.jpg"
  caption: "Generative AI is the most powerful tool for creativity that has ever been created. It has the potential to unleash a new era of human innovation. ~Elon Musk"  
---

In this blog post, we'll delve into the mathematical underpinnings of GPT. Inspired by OpenAI's groundbreaking work, our architecture will incorporate multi-head self-attention and transformer blocks - the core components driving GPT's capabilities. This is  mathematical counterpart of  previous post [here](https://ruslanmv.com/blog/How-to-Build-a-basic-LLM-GPT-from-Scratch-in-Python). 


We will not only cover the mathematical theory but also provide a practical  example. By the end of this post, you will have a solid understanding of how a GPT model processes sequences, how text is generated using a trained GPT model, and how to implement these concepts in Python.

### **Introduction to GPT: The Mathematics Behind Large Language Models**

In this section, we break down the mathematical operations underpinning GPT, enabling us to understand the key steps that allow it to predict and generate human-like text.

---

### **1. Input Sequence as Tokens**

The first step in building a Transformer-based architecture is processing an input sequence. The sequence is typically composed of words or subword tokens, which are discrete and indexed from a vocabulary. Each token is represented as:

$$
X = (x_1, x_2, \dots, x_T)
$$

where $$ T $$ represents the number of tokens in the sequence. Each token $$ x_i $$ corresponds to a word or subword in the sequence. These tokens, in their raw form, are integers mapped from the vocabulary, making them difficult to work with directly in a model that operates on real-valued vectors.

---

### **2. Embedding Layer: Mapping Tokens to Continuous Vectors**

To make the input usable by the model, tokens are transformed into continuous vector representations via an embedding layer. The embedding layer converts each token into a vector in a high-dimensional space, typically with dimensions $$ d_{\text{model}} $$. This process is mathematically expressed as:

$$
Z = (z_1, z_2, \dots, z_T)
$$

where $$ z_i $$ is the embedded representation of the token $$ x_i $$, computed as:

$$
z_i = E(x_i)
$$

Here, $$ E $$ represents the embedding matrix. The embedding layer allows the model to work with continuous vectors, encoding important semantic information about each token while preserving its unique identity.

---

### **3. Transformer Block: Self-Attention Mechanism**

The most critical aspect of the Transformer architecture is its self-attention mechanism, which enables the model to capture dependencies between different tokens in the input sequence. Self-attention allows each token to "attend" to every other token in the sequence, generating a contextualized representation of each token based on its relation to others.

#### **Scaled Dot-Product Attention**

Self-attention can be understood as a mapping from a set of query vectors $$ Q $$, key vectors $$ K $$, and value vectors $$ V $$ to an output. The mathematical formulation for self-attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

In this formula:

- $$ Q $$ is the query matrix (representing the token we're focusing on),
- $$ K $$ is the key matrix (representing the tokens we're attending to),
- $$ V $$ is the value matrix (containing the information we want to extract),
- $$ d_k $$ is the dimensionality of the keys.

The term $$ \frac{QK^T}{\sqrt{d_k}} $$ is used to compute the alignment between different tokens, and the softmax function ensures that the attention scores are normalized to lie between 0 and 1.

Self-attention assigns weights to tokens depending on their relevance to the current token, allowing the model to build a global understanding of the input sequence.

---

### **4. Feed-Forward Neural Network (FFNN)**

After the self-attention mechanism has processed the sequence, each token undergoes further transformation using a feed-forward neural network (FFNN). Unlike the self-attention mechanism, the feed-forward network processes each token independently. The FFNN is defined as:

$$
\text{FFNN}(x) = \sigma(W_1 x + b_1) \times \sigma(W_2 x + b_2)
$$

where:

- $$ W_1 $$ and $$ W_2 $$ are weight matrices,
- $$ b_1 $$ and $$ b_2 $$ are bias vectors,
- $$ \sigma $$ represents a non-linear activation function such as ReLU or GELU.

This network refines the token representations, making them more suitable for subsequent processing by adding non-linearity and complex interactions between features.

---

### **5. Stacking Layers: Building Deep Representations**

The GPT architecture typically consists of multiple layers, each combining self-attention and feed-forward networks. The output of each layer serves as the input to the next layer. This allows the model to build increasingly complex representations of the input sequence as information flows through multiple layers of transformation.

The mathematical representation of a stacked layer can be expressed as:

$$
H^{(l+1)} = \text{LayerNorm}( \text{FFNN}(\text{Attention}(H^{(l)})) + H^{(l)})
$$

Here, $$ H^{(l)} $$ represents the output from the previous layer, and Layer Normalization (LayerNorm) helps stabilize training by normalizing the activations.

---

### **6. Decoder and Output Generation**

GPT uses an autoregressive decoding mechanism, meaning that it generates the next token in the sequence based on the previous tokens. The model processes the final hidden state $$ H $$, which is obtained after stacking multiple attention and feed-forward layers, and projects it back into the vocabulary space:

$$
O = W \times H + b
$$

where $$ W $$ is a weight matrix that projects the hidden states back into the vocabulary space, and $$ b $$ is a bias vector.

The model then applies the softmax function to produce a probability distribution over the vocabulary:

$$
P(y | x) = \text{softmax}(W \times O + b)
$$

This distribution gives the likelihood of each possible next token, allowing the model to generate text one token at a time.

---

### **7. Training Objective: Causal Language Modeling (CLM)**

GPT is trained using a Causal Language Modeling (CLM) objective. In this setup, the model learns to predict the next token in a sequence given the previous tokens. This is done by minimizing the cross-entropy loss between the predicted token probabilities and the actual tokens in the training data:

$$
\mathcal{L} = -\sum_{i=1}^{T} \log P(y_i | x_{1:i-1})
$$

This loss function ensures that the model becomes proficient at predicting the next token in any given sequence, which is crucial for generating coherent text.

---

### **Key Components of the GPT Model in the Code**

Once we've reviewed the basic elements, let us now discuss the components of the GPT model in the Python code and connect them to the mathematical concepts.

---

### **1. Multi-Head Self-Attention Mechanism**

The self-attention mechanism is the foundation of the GPT architecture. It enables the model to generate contextualized representations of each token by allowing tokens to "attend" to other tokens in the input sequence.

#### **Mathematical Formula for Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

Where:

- $$ Q $$ (query), $$ K $$ (key), and $$ V $$ (value) are matrices representing the tokens.
- $$ d_k $$ is the dimensionality of the keys and queries.
- The softmax function normalizes the attention scores.

**Correspondence in Code**:
In the Python implementation, the `MultiHeadSelfAttention` class computes the attention scores between tokens. The attention scores are then used to compute a weighted sum of values for each token, reflecting how much attention a token should give to others:

```python
# Compute energy (QK^T / sqrt(d_k))
energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

# Apply softmax normalization to get attention
attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

# Compute weighted sum of values
out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
```

---

### **2. Positional Encoding**

Transformers are inherently order-agnostic, meaning they do not understand the sequence of tokens unless we explicitly provide positional information. This is achieved through positional encoding, which assigns a unique position to each token in the sequence.

#### **Mathematical Formula for Positional Encoding**:

$$
PE(pos, 2i) = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$
$$
PE(pos, 2i+1) = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

This ensures that each token has a unique position in the sequence.

**Correspondence in Code**:
In the `PositionalEncoding` class, the

 positional encodings are added to the word embeddings to provide the model with information about the position of each token:

```python
encoding[:, 0::2] = torch.sin(pos / (10000 ** (two_i / self.embed_size)))
encoding[:, 1::2] = torch.cos(pos / (10000 ** (two_i / self.embed_size)))
```

These encodings are added to the token embeddings to ensure that the model understands the position of tokens in the sequence.

---

### **3. Transformer Block**

The transformer block is the core computational unit in a GPT model. It combines multi-head self-attention with a feed-forward neural network (FFNN). The FFNN is responsible for refining the token representations generated by the attention mechanism.

#### **Mathematical Formula for FFNN**:

$$
\text{FFNN}(x) = \text{ReLU}(W_1 x + b_1) \times W_2 + b_2
$$

Where:

- $$ W_1 $$ and $$ W_2 $$ are weight matrices.
- $$ b_1 $$ and $$ b_2 $$ are bias terms.
- ReLU is the activation function used to introduce non-linearity.

**Correspondence in Code**:
The `TransformerBlock` class applies multi-head self-attention followed by a feed-forward network:

```python
forward = self.feed_forward(x)
out = self.norm2(forward + x)  # Apply LayerNorm for stability
```

Layer normalization ensures stability during training by normalizing the outputs from each transformer block.

---

### **4. GPT Model Architecture**

A GPT model is a stack of transformer blocks that process input sequences, progressively building richer representations of the tokens. The model generates text by predicting the next token in the sequence based on the previous ones.

#### **Mathematical Formula for GPT**:

$$
P(y_t | x_1, x_2, \dots, x_{t-1}) = \text{softmax}(W \cdot h_t)
$$

Where $$ h_t $$ is the hidden state of the token at position $$ t $$, and $$ W $$ is the projection matrix that maps hidden states back into the vocabulary space.

**Correspondence in Code**:
In the `GPT` class, the forward pass processes the input sequence and generates logits for the next token:

```python
# Get word embeddings and positional encodings
word_embeddings = self.word_embedding(x)
position_encodings = self.position_embedding(word_embeddings)

# Pass through each Transformer block
for layer in self.layers:
    out = layer(out, out, out, mask)

logits = self.fc_out(out)
```

---

### **Practical Example: Generating Text with GPT**

Let's now go over a practical example that ties everything together.

#### **1. Dataset and Tokenization**

To train the GPT model, we need to process a dataset of text. In this case, we use a small sample dataset:

```python
text = """
The quick brown fox jumps over the lazy dog. 
This is an example of a small dataset for training a GPT model.
We are building a transformer-based architecture.
"""
```

**Mathematical Correspondence**:
The text is tokenized into words, and each word is mapped to a numerical index. Mathematically, this corresponds to converting the text into a sequence of tokens:

$$
X = (x_1, x_2, \dots, x_T)
$$

where each token $$ x_i $$ represents a word or subword in the vocabulary.

```python
def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return tokens

def build_vocab(text):
    tokens = tokenize(text)
    vocab = Counter(tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab[token] for token in tokens if token in vocab]
```

The text is broken down into tokens, and a vocabulary is built, mapping each token to a unique index.

---

#### **2. Training the Model**

In the training process, the model is optimized to predict the next token based on the previous tokens in the sequence. During each forward pass, the model computes the probabilities for the next token and compares them with the actual next token in the sequence. The model minimizes the cross-entropy loss:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | x_1, x_2, \dots, x_{t-1})
$$

**Correspondence in Code**:
During training, the model learns to minimize the cross-entropy loss by adjusting its parameters:

```python
for batch in data:
    inputs = batch[:, :-1].to(model.device)
    targets = batch[:, 1:].to(model.device)
    output = model(inputs, mask)

    loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
```

This process is repeated over many epochs to ensure the model learns to predict tokens accurately.

---

#### **3. Example of Text Generation**

Once the model is trained, we can use it to generate text. Given a prompt, the model predicts the next token in the sequence, appends it to the input, and repeats the process until the desired length of text is generated.

Let's consider the prompt:

```python
prompt = "The quick brown"
```

**Mathematical Correspondence**:
The input tokens $$ X = (x_1, x_2, x_3) $$ correspond to the tokens for "The", "quick", and "brown". The model predicts the next token $$ y_4 $$ by computing:

$$
P(y_4 | x_1, x_2, x_3) = \text{softmax}(W \cdot h_3)
$$

Where $$ h_3 $$ is the hidden state after processing "The quick brown". The softmax function converts the logits into probabilities, and the most likely next token is selected.

**Correspondence in Code**:

```python
for _ in range(max_len):
    output = model(inputs, mask)
    next_token_logits = output[0, -1, :]
    predicted_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
    inputs = torch.cat([inputs, predicted_token], dim=1)
```

The model continues generating tokens until the specified maximum length is reached.

---

### **Conclusion**

In this blog post, we've explored the mathematical foundations of building a GPT model from scratch, including multi-head self-attention, positional encoding, and transformer blocks. We also demonstrated how these concepts translate into Python code and applied them to generate text.

By walking through a concrete example of generating text based on a prompt, we've bridged the gap between mathematical theory and practical implementation. With this understanding, you're now ready to explore more advanced concepts in language modeling and GPT architectures!

