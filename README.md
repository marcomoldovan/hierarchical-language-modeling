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