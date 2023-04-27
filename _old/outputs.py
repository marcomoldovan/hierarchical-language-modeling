import torch
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple

class BaseModelOutputWithCrossAttentions(ModelOutput):
  """
  Base class for model's outputs, with potential hidden states and attentions.
  Args:
      last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
          Sequence of hidden-states at the output of the last layer of the model.
      hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
          Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
          of shape :obj:`(batch_size, sequence_length, hidden_size)`.
          Hidden-states of the model at the output of each layer plus the initial embedding outputs.
      attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
          Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
          sequence_length, sequence_length)`.
          Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
          heads.
      cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
          Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
          sequence_length, sequence_length)`.
          Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
          weighted average in the cross-attention heads.
  """

  last_hidden_state: torch.FloatTensor = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None
  cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
  
  
  
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
  """
  Base class for model's outputs that also contains a pooling of the last hidden states.
  Args:
    last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
      Sequence of hidden-states at the output of the last layer of the model.
    pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
      Last layer hidden-state of the first token of the sequence (classification token) further processed by a
      Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
      prediction (classification) objective during pretraining.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.
      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
      sequence_length, sequence_length)`.
      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
    cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
      sequence_length, sequence_length)`.
      Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
      weighted average in the cross-attention heads.
  """
  last_hidden_state: torch.FloatTensor = None
  pooler_output: torch.FloatTensor = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None
  cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
  
  


class SentencePredictionHeadOutput(ModelOutput):
  logits = None
  probabilites = None
  log_probabilities = None
  labels_one_hot = None
  per_example_loss_distance=None
  per_example_loss_product=None
  loss_distance=None
  loss_product=None
  loss_variance_distance=None
  loss_variance_product=None
  
  
  
class SentenceModelingOutput(ModelOutput): #inherits from the huggingface class
  """
    Return object for Sentence Model.

    Args:
      loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
        Masked language modeling (MLM) loss.
      logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
      hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
        Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
        of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
      last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
      attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
        Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
        sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
  """
  loss: Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  last_hidden_state: torch.FloatTensor = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None
  
  
  
  
class DocumentModelingOutput(ModelOutput):
  """
    Return object for Document Model.

    Args:
      loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
        Masked language modeling (MLM) loss.
      logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
      hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
        Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
        of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
      last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
      attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
        Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
        sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
  """
  per_example_loss_distance: Optional[torch.FloatTensor] =None
  per_example_loss_product: Optional[torch.FloatTensor] =None
  loss_distance: Optional[torch.FloatTensor] = None
  loss_product: Optional[torch.FloatTensor] = None
  loss_variance_distance = None
  loss_variance_product = None
  logits: torch.FloatTensor = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  last_hidden_state: torch.FloatTensor = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None
  
  
  
  
class HATEOutput(ModelOutput):
  """
  Class for the whole model output
  """
  sentence_model_hidden_states = None
  document_model_hidden_states = None

  sentence_model_last_hidden_states = None
  document_model_last_hidden_states = None

  sentence_model_attentions = None
  document_model_attentions = None

  sentence_model_logits = None
  document_model_logits = None

  sentence_model_loss = None
  document_model_loss = None
  document_model_per_example_loss = None
  total_loss = None

  loss_variance_distance = None
  loss_variance_product = None
