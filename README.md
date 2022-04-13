# Hierarchical Attention-based Text Encoder (HATE)

We present the multi-purpose Hierarchical Attention-based Text Encoder (HATE), a language
model for learning long-form document representations and contextualized sentence
representations. Our model employs a Transformer module on two separate levels with
the goal of capturing contextual information between words as well as between sentences.
Our model offers theoretical plausibility from a linguistic point of view as we are essentially
trying to model multi-level compositionality. We introduce a pre-training scheme
that uses masked language modeling on two levels in order to train the model to capture
general linguistic features. After that we fine-tune our model on a document ranking
task using the MS MARCO dataset which allows us to compare our performance to many
other models. We detect baseline-beating performance in our fine-tuned models but suspect
severe undertraining. Later in this work we will explore the contextual interactions
and resulting representations visually.

## Hierarchical Modeling for Long-Form Text Representation

### HATE Model Architecture

The HATE model consists of two analogous Transformer encoders with the first one
denoted as the sentence model and the second being the document model. Consider a
document D=(S1,...,Sn) consisting of n sentences. From the point of view of the sentence
model a document acts as a batch which we feed into the model as a tokenized tensor of
input IDs and an attention mask tensor of same shape. Importantly, a CLS token gets
prepended to all sentences.
Since the sentence model is a Transformer encoder which in turn follows the BERT implementation
the input IDs are first passed to the differentiable embedding lookup matrix
and then added to the sinusoidal position embedding, outputting an
uncontexualized word embedding with position information. These vectors serve as the
input into the encoder stack of the sentence model. The encoder stack of the sentence
model closely follows the implementation of BERT but only contains and MLM and no
NSP head.
The sentence model outputs is a 3D tensor of size (Document length, longest sentence
length, hidden size). Consider the first two dimensions given by the matrix
We select the encoded CLS token of every sentence and stack them in a 2D tensor. And
prepend a uniquely selected but random CLS embedding vector at the first position. This
yields the following input sequence for the document model. Note that every entry in the
vector is itself a vector of size 𝑑ℎ𝑖𝑑𝑑𝑒𝑛. Since we are already passing embedding vectors
to our document model we do not pass anything to the embedding lookup matrix and
directly add the positional encoding vector to the input embeddings. The encoder stack
consists of a variable number of encoder modules, each with a Multi-Head Self-Attention
module and an intermediary linear layer with a residual connection between them. Both
the Multi-Head Self-Attention module and the linear layer have a layer normalization
head on top. The linear layer is also followed by a dropout layer. The document model
also follows the BERT implementation but is configured with separate hyperparameters
than the sentence model. It also implements a custom sentence prediction head during
pre-training.
After passing through the encoder stack we can use one of three methods to gather a
document representation:
• CLS Embedding: We simply take the output embedding at the first position as
the document vector, analogous to how the sentence vector is gathered from the
sentence model.
• Mean of Sentence Representations (MSR): We average all the non-padding output
sentence representations and use the resulting vector as the document vector.
• Sum of Sentence Representations (SSR): Same as MSR but instead of averaging
sentence vectors we sum them.

### Downscaling Representations via Autoencoding

Since our research takes place in an academic setting with limited access to compute power
we have decided to introduce a method for reducing the hidden size of our document
model in the hopes of tackling the most significant factor for training speed-up and space
reduction. We are using pre-trained weights for the sentence model for which we keep
these weights fixed, in other words no gradients are computed for the sentence model.
The pre-trained sentence model we used has a hidden size of 128, which stays fixed.
In order to be able to leverage the pre-trained sentence model weights and reduce the
size of the document model we have decided to plug the document model inside an
undercomplete Autoencoder whenever we change the hidden size of the document model.
We have only experimented with undercomplete Autoencoders as increasing the hidden
size in the document model compared to the sentence model did not feel like a sensible
choice.
Whenever HATE is configured to have a document model hidden size smaller than
the sentence model hidden size the outputs from the sentence model are fed into an
encoder with two hidden layers that gradually reduce the dimensionality of the sentence
representations. We have experimented with hidden sizes for the document model of
64, 32 and even 16. The reduced size sentence representations then pass through the
Transformer module and depending on whether the model is in pre-training or fine-tuning
mode the outputs are passed through a decoder. The decoder only gets initialized during
pre-training, the reason for which will become clear when we introduce our pre-training
method in the corresponding section.
The hidden layer sizes of the encoder model are chosen dynamically depending on
the hidden size of the sentence model and the hidden size of the document model. We
compute the intermediary size of the first encoder layer by
intermediary1 = 𝐻𝐷𝑀 + (𝐸𝑙𝑎𝑦𝑒𝑟𝑠) · |step|
and the intermediary size of the second encoder layer by
intermediary2 = 𝐻𝐷𝑀 + (𝐸𝑙𝑎𝑦𝑒𝑟𝑠 − 1) · |step|
where 𝐸𝑙𝑎𝑦𝑒𝑟𝑠 is the number of hidden encoder layers and |𝑠𝑡𝑒𝑝| is the step size given by
|𝑠𝑡𝑒𝑝| = ⌊
𝐻𝑆𝑀 − 𝐻𝐷𝑀
𝐸𝑙𝑎𝑦𝑒𝑟𝑠
⌋
Whenever the decoder is present its intermediary layer sizes are set identically to the
encoder, just in reverse.


## Model Pre-Training

As stated before, we do not consider the sentence model during any training and treat
it as a preprocessing step. We only make the document model weights differentiable and
apply the masked language modeling pre-training paradigm similar to RoBERTa. Here,
instead of masking some input words with a certain probability we mask intermediary
sentence representations coming from the sentence model and let the document model
predict the original uncontextualized sentence representation given its context. Assuming
the model will never learn to overfit the data as much as to perfectly reconstruct
the intermediary sentence representations we assume that the deviations in the output
sentence representations encode some of the document-level context.
There is no global vocabulary of sentences for any given dataset, let alone sentence
vectors. This constitutes the biggest difference between the masked sentence prediction
we are implementing compared to masked word prediction like in BERT or RoBERTa.
Our solution is to construct a dynamic batch-wise vocabulary over which our model will
try to predict masked sentences. Our masking and prediction procedure for unsupervised
masked sentence language modeling are described below.

### Dynamic Sentence Masking

In order to perform masked language modeling on a sentence level we have to mask some
sentence vectors before feeding them into the document model and retain the original
vectors which we consider as the ground truth for the masked sentence prediction.
We closely follow the RoBERTa masking paradigm and do not set fixed masked positions
at the beginning of training. Instead we dynamically sample the masking positions
with a probability of 𝑝 = 0.15. Our sampling algorithm only considers real sentence
embeddings so that we never mask a special token embedding (i.e. CLS, PAD). For every
positively sampled position the masking procedure works according to the following
probabilities:
• 𝑝 = 0.8: The chosen position is replaced with a random but fixed masking vector
• 𝑝 = 0.1: A random vector from the same batch is sampled and replaces the vector
at the masked position. We only consider other real sentence vectors when sampling
from inside the batch.
• 𝑝 = 0.1: We leave the sampled vector as is.
Regardless of what masking is ultimately applied we always store the original sentence
vector for every sampled masking position to use as ground truth during the masked
sentence prediction. We also store a binary tensor of shape (batch size × maximum
document length per batch) with ones at every masked position.

### Masked Sentence Prediction

After having passed the batch of documents with some masked sentences through the document
model we reshape our output tensor from its initial shape of (batch size, maximum
document length per batch, hidden size) to (batch size × maximum document length per
batch, hidden size) in order easily single out the intermediary sentence prediction at the
masked positions with our stored position mask tensor. The resulting tensor will only
contain the stacked intermediary predictions which we pass through another linear layer
and a subsequent layer normalization.
As the tensor ^ 𝑆 containing the predicted sentence vectors and the ground truth label
tensor 𝐿 have the same shape we now calculate the element-wise dot product between
the two matrices. The resulting matrix of dot product similarities is then passed through
a softmax function to output normalized probabilities ℎ. Note the striking similarity to
the attention mechanism
𝑝(ℎ| ^ 𝑆) = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥( ^ 𝑆𝐿𝑇 )
Since the resulting matrix is symmetric with its diagonal containing the probabilities of
the correct predictions we can construct a binary identity matrix 𝐼 of same shape with
1s placed at its diagonal as the target probabilities. We compute the pairwise distance
between the two matrices using the p-norm
𝑥𝑝 = (
Σ︁𝑛
𝑖=1
𝑥𝑖
𝑝)1/𝑝
Resulting in our document model loss function of
ℒ𝐷𝑀 = 1
𝐵
Σ︁𝐵
𝑖=1
Σ︁𝐵
𝑗=1
𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒𝑝𝑎𝑖𝑟𝑤𝑖𝑠𝑒(𝑝(ℎ| ^ 𝑆), 𝐼)
where B is the number of masked sentences per batch. Normally we would backpropagate
from a total
ℒ𝑇𝑜𝑡𝑎𝑙 = ℒ𝑆𝑀 + ℒ𝐷𝑀
where ℒ𝑆𝑀 is the sentence model masked word loss as it is used in BERT and RoBERTa.
But as stated earlier we do not train the sentence model so ℒ𝐷𝑀 remains our pre-training
loss.
