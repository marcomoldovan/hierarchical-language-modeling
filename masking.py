import torch
import transformers
import random
from typing import Tuple, Optional


# word_mask_probability = 0.15
# replace_with_mask_probability = 0.8
# replace_randomly_probability = 0.1
# keep_token_probability = 0.1

def mask_input_ids(inputs: torch.tensor,
                   #tokenizer: transformers.BertTokenizerFast,
                   special_tokens_mask: Optional[torch.Tensor] = None,
                   word_mask_probability = 0.15,
                   replace_with_mask_probability = 0.8,
                   replace_randomly_probability = 0.1,
                   keep_token_probability = 0.1
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  We specifiy the probability with which to mask token for the language modeling
  task. Generally 15% of tokens are considered for masking. If we just mask 
  naively then a problem arises: some masked token will never actually have been 
  seen at fine-tuning. The solution is to not replace the token with [MASK] 100%
  of the time, instead:
  - 80% of the time, replace the token with [MASK]
    went to the store -> went to the [MASK]
  - 10% of the time, replace random token
    went to the store -> went to the running
  - 10% of the time, keep same
    went to the store -> went to the store
  The same principle is also appilicable with masked sentence prediction, only
  that we have to establish a sentence vocabulary beforehand

  Args:
    inputs: tensor, containing all the token IDs
    special_tokens_mask: tensor, denotes whether a token is a word [0] or a 
      special token [1], [CLS] tokens and padding tokens are all counted as 
      special tokens. This will be used to create a mask so that only actual
      words are considered for random masking

  Returns:
    masked_inputs:
    labels:
  """
  labels = inputs.clone()
  # Tensor that hold the probability values for the Bernoulli function
  probability_matrix = torch.full(inputs.shape, word_mask_probability)

  tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

  # Get special token indices in order to exclude special tokens from masking
  if special_tokens_mask is None:
    special_tokens_mask = [
      tokenizer.get_special_tokens_mask(entry, already_has_special_tokens=True) for entry in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
  else:
    special_tokens_mask = special_tokens_mask.bool()

  # Fill the probability matrix with 0.0 values where there are special tokens
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  # Draws from a bernoulli distribution where probability_matrix holds the 
  # probablitites for drawing the binary random number. The probablity matrix
  # was previously filled with 0.0 values where special tokens are present so
  # that only tokens containing words/sentences are considered
  masked_indices = torch.bernoulli(probability_matrix).bool()
  # In order to compute the loss only on the masked indices all the unmasked
  # tokens in the label tensor are set to -100
  labels[~masked_indices] = -100

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, replace_with_mask_probability)).bool() & masked_indices
  # Since we're dealing with tensors with numerical values we convert the [MASK]
  # token right back to its token_id representation
  inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
  inputs[indices_random] = random_words[indices_random]

  return (inputs, labels)



def mask_input_embeddings(input_embeddings: torch.tensor,
                          special_embeddings_mask: torch.tensor,
                          device,
                          sentence_mask_probability = 0.15):
  """
  Randomly masks sentences with a probability of 15%. The masked sentence
  embeddings are replaced with a random tensor and the original embedding will
  be stored in a labels tensor that has the same size as the input tensor. The
  ground truth embedding will sit at the same position as is did in the input
  tensor to make it easier to identify the correct ground truth for loss
  computing.

  Args:
    input_embeddings: A torch.tensor containing all sentence embeddings computed
      by the Sentence Model for a given batch. The size of the tensor is
      [batch_size, max_doc_length, embedding_size]. Note that the documents are
      already padded to the length of the longest document in the batch.
    special_embeddings_mask: A torch.tensor of the same size as input_embeddings
      [batch_size, max_doc_length] which hold 0s where there is a real sentence 
      present and 1s where there is a special token embedding, that includes 
      CLS, SEP and PAD tokens.
  Returns:
    masked_input_embeddings: Same shape as input embeddings, only that it holds
      a random tensor wherever a sentence embedding was masked.
    label_embeddings: Same shape as the masked_input_embeddings but all entries 
      are filled with 0s except where there is a masked sentence embedding. That
      entry will be filled with the original input embedding.
    label_mask: torch.BoolTensor
  """
  masked_input_embeddings = input_embeddings.clone()
  label_embeddings = torch.zeros_like(input_embeddings)
  label_mask = torch.zeros_like(special_embeddings_mask)

  probability_matrix = torch.full(special_embeddings_mask.shape, sentence_mask_probability, device=device)

  probability_matrix.masked_fill_(special_embeddings_mask.bool(), value=0.0)

  masked_indices = torch.bernoulli(probability_matrix).bool()

  # Choose a random index per document to mask in case nothing was randomly masked 
  # via the Bernoulli distribution (will return None, which will lead to an error
  # when we want to manipulate the Tensors inside the loss function)
  if torch.sum(masked_indices.long()).item() == 0:
    forced_mask_indexes = []
    for document in special_embeddings_mask:
      document_list = document.tolist()
      real_indexes = [i for i, x in enumerate(document_list) if x == 0]
      single_choice_per_doc = random.choice(real_indexes)
      forced_mask_indexes.append(single_choice_per_doc)
    for forced_index, previously_masked_doc in zip(forced_mask_indexes, masked_indices):
      previously_masked_doc[forced_index] = True

  document_counter = 0
  sentence_counter = 0

  for document in input_embeddings:
    sentence_counter = 0
    for sentence in document:
      if masked_indices[document_counter][sentence_counter]:
        label_embeddings[document_counter][sentence_counter] = input_embeddings[document_counter][sentence_counter]
        label_mask[document_counter][sentence_counter] = 1.0
        masked_input_embeddings[document_counter][sentence_counter] = torch.randn_like(input_embeddings[document_counter][sentence_counter])
      sentence_counter += 1
    document_counter += 1

  label_embeddings[~masked_indices] = 0
  label_mask = torch.Tensor.bool(label_mask)

  return (input_embeddings, masked_input_embeddings, label_embeddings, label_mask)