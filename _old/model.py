import math

import torch
import torch.nn as nn
import torch.nn.functional as f

import transformers
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.configuration_utils import PretrainedConfig

from outputs import BaseModelOutputWithCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, SentencePredictionHeadOutput, SentenceModelingOutput, DocumentModelingOutput, HATEOutput
from masking import mask_input_ids, mask_input_embeddings


class EmbeddingsLookup(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Initialize the lookup matrix for input IDs, positional embeddings and token types
    self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
    # Adds to Layer Normalization and Dropout on inital word embeddings
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # position_ids (1, len position emb) is contiguous in memory and exported when serialized
    self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
    self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

  def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
    if input_ids is not None:
      input_shape = input_ids.size()
    else:
      input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]

    if position_ids is None:
      position_ids = self.position_ids[:, :seq_length]

    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

    if inputs_embeds is None:
      inputs_embeds = self.word_embeddings(input_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + token_type_embeddings
    if self.position_embedding_type == "absolute":
      position_embeddings = self.position_embeddings(position_ids)
      embeddings += position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings
  
  

class EncoderMLP(nn.Module):
  def __init__(self, config, preembedded_hidden_size):
    super().__init__()

    num_steps = 3
    step_size = (preembedded_hidden_size-config.hidden_size)//num_steps

    intermediary_1 = config.hidden_size + (num_steps-1)*step_size
    intermediary_2 = config.hidden_size + (num_steps-2)*step_size

    self.encoder = nn.Sequential(
        nn.Linear(preembedded_hidden_size, intermediary_1),
        nn.ReLU(),
        nn.Linear(intermediary_1, intermediary_2),
        nn.ReLU(),
        nn.Linear(intermediary_2, config.hidden_size)
    )

  def forward(self, hidden_states):
    outputs = self.encoder(hidden_states)
    return outputs


class DecoderMLP(nn.Module):
  def __init__(self, config, preembedded_hidden_size):
    super().__init__()

    num_steps = 3
    step_size = (preembedded_hidden_size-config.hidden_size)//num_steps

    intermediary_1 = config.hidden_size + (num_steps-1)*step_size
    intermediary_2 = config.hidden_size + (num_steps-2)*step_size

    self.decoder = nn.Sequential(
        nn.Linear(config.hidden_size, intermediary_2),
        nn.ReLU(),
        nn.Linear(intermediary_2, intermediary_1),
        nn.ReLU(),
        nn.Linear(intermediary_1, preembedded_hidden_size)
    )

  def forward(self, hidden_states):
    outputs = self.decoder(hidden_states)
    return outputs
  



class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads)
      )

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
      self.max_position_embeddings = config.max_position_embeddings
      self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self,
              hidden_states,
              attention_mask=None,
              head_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=False):
    
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    if encoder_hidden_states is not None:
        mixed_key_layer = self.key(encoder_hidden_states)
        mixed_value_layer = self.value(encoder_hidden_states)
        attention_mask = encoder_attention_mask
    else:
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)    
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
      seq_length = hidden_states.size()[1]
      position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
      position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
      distance = position_ids_l - position_ids_r
      positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
      positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

      if self.position_embedding_type == "relative_key":
        relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        attention_scores = attention_scores + relative_position_scores
      elif self.position_embedding_type == "relative_key_query":
        relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
      # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
      attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs
  

class SelfOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


class AttentionModule(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self = SelfAttention(config)
    self.output = SelfOutput(config)
    self.pruned_heads = set()

  def prune_head(self, heads):
    if len(heads) == 0:
      return
    heads, index = find_pruneable_heads_and_indices(                            
      heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    )
    # Prune linear layers
    self.self.query = prune_linear_layer(self.self.query, index)
    self.self.key = prune_linear_layer(self.self.key, index)
    self.self.value = prune_linear_layer(self.self.value, index)
    self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    # Update hyper params and store pruned heads
    self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  def forward(self,
              hidden_states,
              attention_mask=None,
              head_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=False):
    self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions)
    attention_output = self.output(self_outputs[0], hidden_states)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs





class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    if isinstance(config.hidden_act, str):
      self.intermediate_act_fn = ACT2FN[config.hidden_act]
    else:
      self.intermediate_act_fn = config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


class Output(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states





class EncoderLayer (nn.Module):
  def __init__(self, config):
    super().__init__()
    self.chunk_size_feed_forward = config.chunk_size_feed_forward
    self.seq_len_dim = 1
    self.attention = AttentionModule(config)
    self.is_decoder = config.is_decoder
    self.add_cross_attention = config.add_cross_attention
    if self.add_cross_attention:
      assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
      self.crossattention = AttentionModule(config)
    self.intermediate = FeedForward(config)
    self.output = Output(config)

  def forward(self,
              hidden_states,
              attention_mask=None,
              head_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=False):
    
    self_attention_outputs = self.attention(hidden_states,
                                            attention_mask,
                                            head_mask,
                                            output_attentions=output_attentions)
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
    if self.is_decoder and encoder_hidden_states is not None:
      assert hasattr(self, "crossattention"), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
      cross_attention_outputs = self.crossattention(attention_output,
                                                    attention_mask,
                                                    head_mask,
                                                    encoder_hidden_states,
                                                    encoder_attention_mask,
                                                    output_attentions)
      attention_output = cross_attention_outputs[0]
      outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

    layer_output = apply_chunking_to_forward(self.feed_forward_chunk,           # don't forget this!!
                                             self.chunk_size_feed_forward, 
                                             self.seq_len_dim, attention_output)
      
    outputs = (layer_output,) + outputs
    return outputs

  def feed_forward_chunk(self, attention_output):
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output




class EncoderStack(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self,
              hidden_states,
              attention_mask=None,
              head_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=False,
              output_hidden_states=False,
              return_dict=True):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    for i, layer_module in enumerate(self.layer):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
      
      layer_head_mask = head_mask[i] if head_mask is not None else None

      if getattr(self.config, "gradient_checkpointing", False):
        def create_custom_forward(module):
          def custom_forward(*inputs):
            return module(*inputs, output_attentions)
          return custom_forward
        layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module),
                                                          hidden_states,
                                                          attention_mask,
                                                          layer_head_mask,
                                                          encoder_hidden_states,
                                                          encoder_attention_mask,)
      else:
        layer_outputs = layer_module(hidden_states,
                                     attention_mask,
                                     layer_head_mask,
                                     encoder_hidden_states,
                                     encoder_attention_mask,
                                     output_attentions,)
      
      hidden_states = layer_outputs[0]
      if output_attentions:
        all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if self.config.add_cross_attention:
          all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(v
                   for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                   if v is not None)
    
    return BaseModelOutputWithCrossAttentions(last_hidden_state=hidden_states,
                                              hidden_states=all_hidden_states,
                                              attentions=all_self_attentions,
                                              cross_attentions=all_cross_attentions,)
    

class Pooler(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = nn.Tanh()

  def forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output



class TransformerPreTrainedModel(PreTrainedModel):
  """
  An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
  models.
  """
  config_class = BertConfig
  base_model_prefix = "bert"
  _keys_to_ignore_on_load_missing = [r"position_ids"]

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
      
      


class TransformerBase(BertPreTrainedModel):
  """
  The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
  cross-attention is added between the self-attention layers, following the architecture described in `Attention is
  all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
  Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
  To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
  set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
  argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
  input to the forward pass.
  """
  def __init__(self, config, add_pooling_layer=True):
    super().__init__(config)
    self.config = config

    self.embeddings = EmbeddingsLookup(config)
    self.encoder = EncoderStack(config)

    self.pooler = Pooler(config) if add_pooling_layer else None

    self.init_weights() # Don't forget this

  def get_input_embeddings(self):
    return self.embeddings.word_embeddings

  def set_input_embeddings(self, value):
    self.embeddings.word_embeddings = value

  def _prune_heads(self, heads_to_prune):
    """
    Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    class PreTrainedModel
    """
    for layer, heads in heads_to_prune.items():
      self.encoder.layer[layer].attention.prune_heads(heads)

  def forward(self,
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=None,
              output_hidden_states=None,
              return_dict=None,):
    """
    encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
      Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
      the model is configured as a decoder.
    encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
      Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
      the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
      - 1 for tokens that are **not masked**,
      - 0 for tokens that are **masked**.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
      raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      input_shape = input_ids.size()
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
      attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device) # can we throw this away?

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
      encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
      encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
      if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
      encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
      encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(input_ids=input_ids,
                                       position_ids=position_ids,
                                       token_type_ids=token_type_ids,
                                       inputs_embeds=inputs_embeds)
    encoder_outputs = self.encoder(embedding_output,
                                   attention_mask=extended_attention_mask,
                                   head_mask=head_mask,
                                   encoder_hidden_states=encoder_hidden_states,
                                   encoder_attention_mask=encoder_extended_attention_mask,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
      return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output,
                                                        pooler_output=pooled_output,
                                                        hidden_states=encoder_outputs.hidden_states,
                                                        attentions=encoder_outputs.attentions,
                                                        cross_attentions=encoder_outputs.cross_attentions,)
    
    


class PredictionHeadTransform(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    if isinstance(config.hidden_act, str):
      self.transform_act_fn = ACT2FN[config.hidden_act]
    else:
      self.transform_act_fn = config.hidden_act
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.LayerNorm(hidden_states)
    return hidden_states



class LMPredictionHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.transform = PredictionHeadTransform(config)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
    self.decoder.bias = self.bias

  def forward(self, hidden_states):
    hidden_states = self.transform(hidden_states)
    hidden_states = self.decoder(hidden_states)
    return hidden_states



class OnlyMLMHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.predictions = LMPredictionHead(config)

  def forward(self, sequence_output):
    prediction_scores = self.predictions(sequence_output)
    return prediction_scores
  
  
  
  
class SentencePredictionHead(nn.Module):
  def __init__(self, config, preembedded_hidden_size):
    super().__init__()
    self.dense = nn.Linear(in_features=preembedded_hidden_size, out_features=preembedded_hidden_size)
    self.LayerNorm = nn.LayerNorm(preembedded_hidden_size, eps=config.layer_norm_eps) # Compare to the word-level LM Head
    self.config = config

  def forward(self, masked_sentence_prediction, label_embeddings, label_mask):
    """
    In order to compute the sentence-level prediction loss we apply a similar
    loss function as during the word-level masked word prediction tast. Since
    we don't have a fixed size vocabulary over the training sentences we have
    to build a dynamic sentence vocabulary.
    Args:
      masked_sentence_prediction [batch_size, max_doc_length, hidden_size]:
      label_embeddings [batch_size, max_doc_length, hidden_size]:
      label_mask [batch_size, max_doc_length, hidden_size]:
    Returns:
      per_batch_sentence_loss:
      per_example_sentence_loss:
    """
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    label_mask = torch.Tensor.bool(label_mask)
    predictions = masked_sentence_prediction.clone()
    # Zero out all sentence embeddings that aren't at a masked position
    predictions[~label_mask] = 0.0
    label_embeddings[~label_mask] = 0.0
    
    # Tensors will have size [batch_size * padded_doc_length, hidden_size]
    predictions = torch.reshape(predictions, (predictions.size()[0] * predictions.size()[1], -1)) 
    label_embeddings = torch.reshape(label_embeddings, (label_embeddings.size()[0] * label_embeddings.size()[1], -1))
    label_mask = torch.reshape(label_mask, (label_mask.size()[0] * label_mask.size()[1], -1))

    output_embedding_list = []
    label_embedding_list = []
    label_mask_index =  0

    for mask_index in label_mask:
      if mask_index.item():
        output_embedding_list.append(predictions[label_mask_index])
        label_embedding_list.append(label_embeddings[label_mask_index])
      label_mask_index += 1

    output_embeddings = torch.stack(output_embedding_list, dim=0)
    label_embeddings = torch.stack(label_embedding_list, dim=0)


    output_embeddings = self.dense(output_embeddings)
    output_embeddings = self.LayerNorm(output_embeddings)

    # TODO add bias like in SMITH? (smith/layers.py)

    logits = torch.matmul(output_embeddings, torch.transpose(input=label_embeddings, dim0=0, dim1=1))
    probabilities = f.softmax(logits, dim=1)
    log_probabilities = f.log_softmax(logits, dim=1)
    labels_one_hot = torch.diag(torch.Tensor([1] * log_probabilities.size()[0]))
    labels_one_hot = labels_one_hot.to(device)

    # Computes the pairwise distance between each row of the inputs, meaning that
    # it computes the elementwise difference between probabilities and labels and
    # sums them up per one prediction (row).
    per_example_loss_distance = f.pairwise_distance(probabilities, labels_one_hot, p=1.0, keepdim=False)
    # Another option is to compute the pairwise product per element of the log_probs
    # and the one hot labels and sum them per prediction (romw). This emphasizes
    # very bad predictions less and keeps the loss smaller.
    per_example_loss_product = -torch.sum(log_probabilities * labels_one_hot, 1)

    #Shape: [1]
    numerator_distance = torch.sum(per_example_loss_distance)
    numerator_product = torch.sum(per_example_loss_product)

    # Shape: [1], small fraction added so we never divide by 0
    denominator =  labels_one_hot.size()[0] + 1e-5

    # Shape: [1]
    loss_distance = numerator_distance / denominator
    loss_product = numerator_product / denominator

    # Shape: [1]
    loss_variance_distance = torch.var(per_example_loss_distance)
    loss_variance_product = torch.var(per_example_loss_product)

    return SentencePredictionHeadOutput(logits=logits,
                                        probabilites=probabilities,
                                        log_probabilities=log_probabilities,
                                        labels_one_hot=labels_one_hot,
                                        per_example_loss_distance=per_example_loss_distance,
                                        per_example_loss_product=per_example_loss_product,
                                        loss_distance=loss_distance,
                                        loss_product=loss_product,
                                        loss_variance_distance=loss_variance_distance,
                                        loss_variance_product=loss_variance_product)
    
    
    
    
  
class HATESentenceModel(TransformerPreTrainedModel):
  
  _keys_to_ignore_on_load_unexpected = [r"pooler"]
  _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

  
  def __init__(self, config, pretrain_sentence_model):
    super().__init__(config)

    if config.is_decoder:
      print(
        "If you want to use this model for masked LM make sure `config.is_decoder=False` for "
        "bi-directional self-attention."
        )
      
    if pretrain_sentence_model:
      self.transformer = TransformerBase(config, add_pooling_layer=False)
    else:
      self.transformer = transformers.BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
      for param in self.transformer.base_model.parameters():
        param.requires_grad = False

    self.cls = OnlyMLMHead(config)


  def get_output_embeddings(self):
    return self.cls.predictions.decoder

  def set_output_embeddings(self, new_embeddings):
    self.cls.predictions.decoder = new_embeddings

  def forward(self,
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              labels=None,
              output_attentions=None,
              output_hidden_states=None,
              return_dict=None,):
    # TODO replace batch_size with document_length in here in the docfile
    """
    Args:
      inputs_ids (torch.LongTensor of shape (batch_size, sequence_length)):
        Indices of input sequence tokens in the vocabulary.
      attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        - 1 for tokens that are not masked,
        - 0 for tokens that are masked.
      token_type_ids  (torch.LongTensor of shape (batch_size, sequence_length), optional):
        Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
        - 0 corresponds to a sentence A token,
        - 1 corresponds to a sentence B token.
      position_ids (torch.LongTensor of shape (batch_size, sequence_length), optional):
        Indices of positions of each input sequence tokens in the position embeddings. 
        Selected in the range [0, config.max_position_embeddings - 1].
      head_mask (torch.FloatTensor of shape (num_heads,) or (num_layers, num_heads), optional):
        Mask to nullify selected heads of the self-attention modules. Mask values selected in [0, 1]:
        - 1 indicates the head is not masked,
        - 0 indicates the head is masked.
      inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
        Optionally, instead of passing input_ids you can choose to directly pass
         an embedded representation. This is useful if you want more control over 
         how to convert input_ids indices into associated vectors than the model’s 
         internal embedding lookup matrix.
      encoder_hidden_states (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
        Sequence of hidden-states at the output of the last layer of the encoder. 
        Used in the cross-attention if the model is configured as a decoder.
      encoder_attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
        Mask to avoid performing attention on the padding token indices of the encoder 
        input. This mask is used in the cross-attention if the model is configured 
        as a decoder. Mask values selected in [0, 1]:
        - 1 for tokens that are not masked,
        - 0 for tokens that are masked.
      labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
        config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
        (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
      output_attentions (bool, optional): 
        Whether or not to return the attentions tensors of all attention layers. 
        See attentions under returned tensors for more detail.
      output_hidden_states (bool, optional):
        Whether or not to return the hidden states of all layers. See hidden_states 
        under returned tensors for more detail.
      return_dict (bool, optional):
        Whether or not to return a ModelOutput instead of a plain tuple.
    Returns:
      SentenceModelingOutput:
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.transformer(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)
    
    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)

    masked_lm_loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
      masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
      output = (prediction_scores,) + outputs[2:]
      return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return SentenceModelingOutput(loss=masked_lm_loss,
                                  logits=prediction_scores,
                                  hidden_states=outputs.hidden_states,
                                  last_hidden_state=sequence_output,
                                  attentions=outputs.attentions,)
    
    
    

class HATEDocumentModel(TransformerPreTrainedModel):

  _keys_to_ignore_on_load_unexpected = [r"pooler"]
  _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

  def __init__(self, config, preembedded_hidden_size=128):
    super().__init__(config)

    self.preembedded_hidden_size = preembedded_hidden_size
    self.config = config

    if not preembedded_hidden_size == config.hidden_size:
      self.encoder = EncoderMLP(config, preembedded_hidden_size)
      self.decoder = DecoderMLP(config, preembedded_hidden_size)

    self.transformer = TransformerBase(config, add_pooling_layer=False)
    self.cls = SentencePredictionHead(config, preembedded_hidden_size)
    
  def forward(self,
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              labels_embeddings=None,
              labels_mask=None,
              output_attentions=None,
              output_hidden_states=None,
              return_dict=None,):
    # TODO replace batch_size with document_length in here in the docfile
    """
    Args:
      inputs_ids (torch.LongTensor of shape (batch_size, sequence_length)):
        Indices of input sequence tokens in the vocabulary.
      attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        - 1 for tokens that are not masked,
        - 0 for tokens that are masked.
      token_type_ids  (torch.LongTensor of shape (batch_size, sequence_length), optional):
        Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
        - 0 corresponds to a sentence A token,
        - 1 corresponds to a sentence B token.
      position_ids (torch.LongTensor of shape (batch_size, sequence_length), optional):
        Indices of positions of each input sequence tokens in the position embeddings. 
        Selected in the range [0, config.max_position_embeddings - 1].
      head_mask (torch.FloatTensor of shape (num_heads,) or (num_layers, num_heads), optional):
        Mask to nullify selected heads of the self-attention modules. Mask values selected in [0, 1]:
        - 1 indicates the head is not masked,
        - 0 indicates the head is masked.
      inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
        Optionally, instead of passing input_ids you can choose to directly pass
         an embedded representation. This is useful if you want more control over 
         how to convert input_ids indices into associated vectors than the model’s 
         internal embedding lookup matrix.
      encoder_hidden_states (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
        Sequence of hidden-states at the output of the last layer of the encoder. 
        Used in the cross-attention if the model is configured as a decoder.
      encoder_attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
        Mask to avoid performing attention on the padding token indices of the encoder 
        input. This mask is used in the cross-attention if the model is configured 
        as a decoder. Mask values selected in [0, 1]:
        - 1 for tokens that are not masked,
        - 0 for tokens that are masked.
      labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
        config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
        (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
      output_attentions (bool, optional): 
        Whether or not to return the attentions tensors of all attention layers. 
        See attentions under returned tensors for more detail.
      output_hidden_states (bool, optional):
        Whether or not to return the hidden states of all layers. See hidden_states 
        under returned tensors for more detail.
      return_dict (bool, optional):
        Whether or not to return a ModelOutput instead of a plain tuple.
    Returns:
      DocumentModelingOutput:
    """

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if not self.preembedded_hidden_size == self.config.hidden_size:
      inputs_embeds = self.encoder(inputs_embeds)

    outputs = self.transformer(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,)
    last_hidden_state = outputs['last_hidden_state']
    
    if labels_embeddings is not None:
      if not self.preembedded_hidden_size == self.config.hidden_size:
        last_hidden_state = self.decoder(last_hidden_state)

      sentence_prediction_output = self.cls(last_hidden_state, labels_embeddings, labels_mask)

      return DocumentModelingOutput(per_example_loss_distance=sentence_prediction_output[4],
                                    per_example_loss_product=sentence_prediction_output[5],
                                    loss_distance=sentence_prediction_output[6],
                                    loss_product=sentence_prediction_output[7],
                                    loss_variance_distance=sentence_prediction_output[8],
                                    loss_variance_product=sentence_prediction_output[9],
                                    logits=sentence_prediction_output[0],
                                    hidden_states=outputs.hidden_states,
                                    last_hidden_state=last_hidden_state,
                                    attentions=outputs.attentions,)
    
    else:
      return DocumentModelingOutput(hidden_states=outputs.hidden_states,
                                    last_hidden_state=last_hidden_state,
                                    attentions=outputs.attentions,)
      
      
      
      
class HATEDocumentModelCrossEncoderRanking(TransformerPreTrainedModel):

  def __init__(self, config, preembedded_hidden_size=128, ranking_loss_margin=0.25):
    super().__init__(config)

    self.preembedded_hidden_size = preembedded_hidden_size
    self.config = config
    self.ranking_loss_margin = ranking_loss_margin

    if not preembedded_hidden_size == config.hidden_size:
      self.encoder = EncoderMLP(config, preembedded_hidden_size)

    self.transformer = TransformerBase(config, add_pooling_layer=False)
    self.cls = nn.Linear(config.hidden_size, 1)

  def forward(self,
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=None,
              output_hidden_states=None,
              compute_loss=False,
              labels_initial=None,
              query_ids=None, # for the "hardmode" finetuning where we want to distinguish more negative samples from a first stage retrieval but we have to know what query all of them belong to
              return_dict=None,):
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if not self.preembedded_hidden_size == self.config.hidden_size:
      inputs_embeds = self.encoder(inputs_embeds)

    outputs = self.transformer(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,)
    

    cls_embeddings = []
    for query in outputs['last_hidden_state']:
      cls_embeddings.append(query[0])
    cls_embeddings = torch.stack(cls_embeddings)

    predictions = torch.sigmoid(self.cls(cls_embeddings)) # Instead of linear layer use pooling layer (similar to SMITH)

    if self.training:
      if query_ids is not None and labels is not None:
        print("TODO")
      else:
        loss_fct = nn.MarginRankingLoss(margin=self.ranking_loss_margin, reduction='sum')

        split_size = int((predictions.size()[0])/2)

        predictions_split = torch.split(predictions, split_size)

        x = torch.cat((predictions_split[0], predictions_split[1]))
        y = torch.cat((predictions_split[1], predictions_split[0]))

        labels_pos = torch.ones_like(predictions_split[0])
        labels_neg = torch.neg(labels_pos)
        labels = torch.cat((labels_pos, labels_neg))

        loss = loss_fct(x, y, labels)

      return cls_embeddings, predictions, loss
    
    elif not self.training and labels_initial is not None:
      predictions = torch.flatten(predictions)
      sorted_predictions, sorted_indices = torch.sort(predictions, descending=True)
      labels_initial = torch.flatten(labels_initial)
      labels_reranked = labels_initial[sorted_indices]

      return cls_embeddings, predictions, sorted_predictions, sorted_indices, labels_initial, labels_reranked
    
    
    
    
    
class HATEDocumentModelCosineSimilarityRanking(TransformerPreTrainedModel):
  def __init__(self, config, preembedded_hidden_size=128, ranking_loss_margin=0.25):
    super().__init__(config)

    self.preembedded_hidden_size = preembedded_hidden_size
    self.config = config
    self.ranking_loss_margin = ranking_loss_margin

    if not preembedded_hidden_size == config.hidden_size:
      self.encoder = EncoderMLP(config, preembedded_hidden_size)
    
    self.transformer = TransformerBase(config, add_pooling_layer=False)

  def forward(self,
              embedding_mode='cls',
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              query_embeds=None,
              inputs_embeds=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None,
              output_attentions=None,
              output_hidden_states=None,
              compute_loss=False,
              labels_initial=None,
              query_ids=None, # for the "hardmode" finetuning where we want to distinguish more negative samples from a first stage retrieval but we have to know what query all of them belong to
              return_dict=None,):
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if not self.preembedded_hidden_size == self.config.hidden_size:
      query_embeds = self.encoder(query_embeds)
      inputs_embeds = self.encoder(inputs_embeds)

    outputs = self.transformer(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,)
    

    doc_embeddings = []
    
    for i, doc in enumerate(outputs['last_hidden_state']):
      if embedding_mode == 'cls':
        doc_embeddings.append(doc[0])
      elif embedding_mode == 'mean':
        doc_not_as_view = torch.clone(doc)
        doc_not_as_view[0] = 0
        doc_not_as_view[~attention_mask[i].bool()] = 0
        doc_embed = torch.mean(doc_not_as_view, dim=0)
        doc_embeddings.append(doc_embed)
      elif embedding_mode == 'sum':
        doc_not_as_view = torch.clone(doc)
        doc_not_as_view[0] = 0
        doc_not_as_view[~attention_mask[i].bool()] = 0
        doc_embed = torch.sum(doc_not_as_view, dim=0)
        doc_embeddings.append(doc_embed)

    doc_embeddings = torch.stack(doc_embeddings)  # Add pooling pooling layer for better doc reps (similar to SMITH)

    if self.training:
      loss_fct = nn.CosineEmbeddingLoss(margin=self.ranking_loss_margin, reduction='sum') # #0.25 worked somewhat TODO try torch.nn.HingeEmbeddingLoss()
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      shuffled_indexes = torch.randperm(doc_embeddings.shape[0])
      doc_shuffled = doc_embeddings[shuffled_indexes]

      queries_stacked = torch.cat((query_embeds, query_embeds))
      doc_stacked = torch.cat((doc_embeddings, doc_shuffled))

      labels_pos = torch.ones(query_embeds.size()[0], device=device)
      labels_neg = torch.neg(labels_pos)
      labels_stacked = torch.cat((labels_pos, labels_neg))

      predictions = f.cosine_similarity(queries_stacked, doc_stacked)

      loss = loss_fct(queries_stacked, doc_stacked, labels_stacked)

      return doc_embeddings, predictions, loss
    
    elif not self.training and labels_initial is not None:
      query_embeds = torch.squeeze(query_embeds, dim=1)
      predictions = f.cosine_similarity(query_embeds, doc_embeddings)

      sorted_predictions, sorted_indices = torch.sort(predictions, descending=True)
      labels_reranked = labels_initial[0][sorted_indices]

      return doc_embeddings, predictions, sorted_predictions, sorted_indices, labels_initial, labels_reranked
    
    
    
    
class NaiveCosineSimilarityRanking(TransformerPreTrainedModel):
  def __init__(self, config, pretrained_transformer_weights=None, finetuned_model_weights=None):
    super().__init__(config)

  def forward(self,
              doc_embedding,
              attention_mask=None,
              query_embeds=None,
              inputs_embeds=None,
              labels_initial=None):
    
    # doc_embedding can be either 'mean' or 'sum'

    doc_embeds = []
    for i, doc in enumerate(inputs_embeds):
      doc[0] = 0
      doc[~attention_mask[i].bool()] = 0
      if doc_embedding == 'mean':
        doc_embed = torch.mean(doc, dim=0)
        doc_embeds.append(doc_embed)
      elif doc_embedding == 'sum':
        doc_embed = torch.sum(doc, dim=0)
        doc_embeds.append(doc_embed)

    query_embeds = torch.squeeze(query_embeds, dim=1)
    doc_embeds = torch.stack(doc_embeds)
    predictions = f.cosine_similarity(query_embeds, doc_embeds)

    sorted_predictions, sorted_indices = torch.sort(predictions, descending=True)
    labels_reranked = labels_initial[0][sorted_indices]

    return predictions, sorted_predictions, sorted_indices, labels_initial, labels_reranked
  
  
  
  
class HATEModel (torch.nn.Module):
  def __init__(self, hate_config):
    super().__init__()

    self.hate_config = hate_config

    
    self.sentence_model = HATESentenceModel(hate_config.sentence_model_config, 
                                            hate_config.pretrain_sentence_model)
    self.document_model = HATEDocumentModel(hate_config.document_model_config, 
                                            hate_config.pretrain_document_model)
    
    
    # Intermediate layers could be trained if hiden_size changes between models
    # self.intermediate_dense = nn.Linear(in_features=hate_config.sentence_model_config.hidden_size,
    #                                     out_features=hate_config.document_model_config.hidden_size)
    # self.intermediary_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    # self.intermediary_dropout =  nn.Dropout()


  def forward(self, max_doc_length, doc_length_before_padding, batch, device): 
    """
    Args:
      batch: List with 3 entries from our custom collate function
        batch[0]: The length of the largest document in the batch, the other
          documents in the batch will be padded to this length with 0-sentences 
        batch[1]: A list of the original unpadded size of every document in the
          batch, will be used when the intermediary sentence representations are
          padded with the appropriate attention mask and special token mask for
          the document model.
        batch[2]: Dictionary that contains the padded tensors.
          ['input_ids'] (batch_size, max_doc_length, padding_strategy)
          ['special_tokens_mask'] (batch_size, max_doc_length, padding_strategy)
          ['attention_mask'] (batch_size, max_doc_length, padding_strategy)
          ['token_type_ids'] (batch_size, max_doc_length, padding_strategy)
    Returns:
      A HATEModelOutput Object, entries are filled depending on whether model is
      in pretraining mode or not. During pretraining every entry is filled. If
      model is not in pretraining it won't return losses and logits.
    """

    if self.hate_config.sentence_model_config.hidden_size != self.hate_config.sentence_model_config.hidden_size:
      raise Exception(f"Hidden sizes of Sentence and Document Model must be compatibe, got {self.hate_config.sentence_model_config.hidden_size} and {self.hate_config.sentence_model_config.hidden_size}")

    sentence_level_loss = 0
    document_level_loss = 0

    # Initiates lists that aggregate input for the document model
    intermediary_embeddings = []
    intermediary_attention_mask = []
    intermediary_special_tokens_mask = []

    # Initiates lists later used in HATEModelOutput
    sentence_model_hidden_states = []
    sentence_model_last_hidden_states = []
    sentence_model_attentions = []
    sentence_model_logits = []

    # Iterate over documents and apply the Sentence Model per doc
    for doc_counter, doc in enumerate(batch):
      
      # Initiate embedding tensor, attention mask and special tokens mask for one
      # document. The individual sentence embeddings and their according attention
      # mask label as well as special token label will be appended to this. We 
      # initiate them with a random tensor at position 0 to denote the CLS token.
      sentence_embeddings_per_doc = [torch.randn(self.hate_config.hidden_size, device=device)]
      attention_mask_per_doc = [1]
      special_tokens_mask_per_doc = [1]

      # Compute sentence loss only if (sentence) model is training
      if self.hate_config.is_pretraining and self.hate_config.pretrain_sentence_model:
        # Mask input IDs for one document
        masking_output = mask_input_ids(doc['input_ids'], doc['special_tokens_mask'])
        # Feed all the inputs to the Sentence Model for one document
        sentence_model_output = self.sentence_model(input_ids=masking_output[0], 
                                                    attention_mask=doc['attention_mask'], 
                                                    token_type_ids=doc['token_type_ids'],
                                                    labels=masking_output[1],
                                                    output_attentions=True,
                                                    output_hidden_states=True)
        sentence_level_loss += sentence_model_output['loss']
        
      # If either of the pretrain settings is set to false we don't compute loss
      elif self.hate_config.is_pretraining == False or self.hate_config.pretrain_sentence_model == False:
        sentence_model_output = self.sentence_model(input_ids=doc['input_ids'], 
                                                    attention_mask=doc['attention_mask'], 
                                                    token_type_ids=doc['token_type_ids'],)
      
      # TODO fix this so we have a proper output
      # Aggregate releveant sentence model outputs for HATEModelOutput
      sentence_model_hidden_states.append(sentence_model_output['hidden_states'])
      sentence_model_last_hidden_states.append(sentence_model_output['last_hidden_state'])
      sentence_model_attentions.append(sentence_model_output['attentions'])
      sentence_model_logits.append(sentence_model_output['logits'])

      # Iterate over sequence embeddings returned by the model for one document
      # sentence_model_output['last_hidden_state'] has size 
      # (doc_length, max_position_embeddings, hidden_size)
      for sentence_counter, sentence in enumerate(sentence_model_output['last_hidden_state']):
        # CLS embedding for a sentence at sentence_counter position in the document
        # CLS is at position 0 out of max_position_embeddings (512 usually)
        # Pad the embeddings and additional necessary inputs to the maximum document length in the batch
        sentence_embeddings_per_doc.append(sentence[0])
        # If the sentence representations are still from 'real' sentences then we
        # add positive attention mask and negative special token mask
        if sentence_counter < doc_length_before_padding[doc_counter]:
          attention_mask_per_doc.append(1)
          special_tokens_mask_per_doc.append(0)
        # If the sentence representation is just a padding added by our custom
        # collate function then we add the masks the other way around.
        else:
          attention_mask_per_doc.append(0)
          special_tokens_mask_per_doc.append(1)


      # TODO insert linear layer+layernorm+dropout in case sentence_model.hidden_size != document_model.hidden_size in order to transform the embeddings


      # Appends the padded tensors to the list of document-wise sentence embeddings
      intermediary_embeddings.append(torch.stack(sentence_embeddings_per_doc))
      if torch.cuda.is_available():
        intermediary_attention_mask.append(torch.cuda.FloatTensor(attention_mask_per_doc))
        intermediary_special_tokens_mask.append(torch.cuda.FloatTensor(special_tokens_mask_per_doc))
      else:
        intermediary_attention_mask.append(torch.FloatTensor(attention_mask_per_doc))
        intermediary_special_tokens_mask.append(torch.FloatTensor(special_tokens_mask_per_doc))

    # Stacks the list into a Torch Tensor of fixed size
    # Embedding tensor has size (batch_size, max_doc_length+1, hidden_size)
    intermediary_embeddings_tensor = torch.stack(intermediary_embeddings)
    intermediary_attention_mask_tensor = torch.stack(intermediary_attention_mask)
    intermediary_special_tokens_mask_tensor = torch.stack(intermediary_special_tokens_mask)

    # TODO add dense layer and normalization on intermediary_embeddings_tensor?

    # Don't perform masking if pretraining=False:
    if self.hate_config.is_pretraining:
      masked_input_embeddings = mask_input_embeddings(intermediary_embeddings_tensor, 
                                                      intermediary_special_tokens_mask_tensor,
                                                      device)

      document_model_output = self.document_model(attention_mask=intermediary_attention_mask_tensor,
                                                  inputs_embeds=masked_input_embeddings[1],
                                                  labels_embeddings=masked_input_embeddings[2],
                                                  labels_mask=masked_input_embeddings[3],)
      

      # Allow switching between loss functions in document model
      if self.hate_config.use_product_loss:
        document_level_loss += document_model_output['loss_product']
        document_model_per_example_loss = document_model_output['per_example_loss_product']
      else:
        document_level_loss += document_model_output['loss_distance']
        document_model_per_example_loss = document_model_output['per_example_loss_distance']

      total_loss = sentence_level_loss + document_level_loss
      if total_loss.item() == 0.0:
        # TODO Seems to appear when only a singe document in the batch is masked
        print("Weird loss\nSentence loss: ", sentence_level_loss, "\nDocument loss: ", document_level_loss)
        print("Intermediary embeddings: ", intermediary_embeddings_tensor)
        print("Embeddings masking: ", masked_input_embeddings)

      return HATEOutput(sentence_model_hidden_states=sentence_model_hidden_states,
                        document_model_hidden_states=document_model_output['hidden_states'],
                        sentence_model_last_hidden_states=sentence_model_last_hidden_states,
                        document_model_last_hidden_states=document_model_output['last_hidden_state'],
                        sentence_model_attentions=sentence_model_attentions,
                        document_model_attentions=document_model_output['attentions'],
                        sentence_model_logits=sentence_model_logits,
                        document_model_logits=document_model_output['logits'],
                        sentence_model_loss=sentence_level_loss,
                        document_model_loss=document_level_loss,
                        document_model_per_example_loss=document_model_per_example_loss,
                        total_loss=total_loss,
                        loss_variance_distance=document_model_output['loss_variance_distance'],
                        loss_variance_product=document_model_output['loss_variance_product'])
    else:
      document_model_output = self.document_model(attention_mask=intermediary_attention_mask_tensor,
                                             inputs_embeds=intermediary_embeddings_tensor)
      
      return HATEOutput(sentence_model_hidden_states=sentence_model_hidden_states,
                        document_model_hidden_states=document_model_output['hidden_states'],
                        sentence_model_last_hidden_states=sentence_model_last_hidden_states,
                        document_model_last_hidden_states=document_model_output['last_hidden_state'],
                        sentence_model_attentions=sentence_model_attentions,
                        document_model_attentions=document_model_output['attentions'])

    # TODO prune layers? transformers.modeling_utils.find_pruneable_heads_and_indices
    
    
    
    
class HATEModelForDocumentRanking(torch.nn.Module):
  def __init__():
    print("TODO")
    
  def forward(first_stage_retrievals=None, labels=None, is_training=False, use_cosine_similarity=False, use_cross_encoder=False): 
    # TODO put is_training in config
    # TODO as a matter of fact put as much as possible in the config, only leave data handover to the forward() arguments
    
    # Regular sentence model

    # TODO both of the following should work for training and inference
    if labels is None:
      # Assume all samples are positives
      # Feed into document model similar to prepare_for_msmarco_cross_encoder/_cosine_similarity
      if use_cosine_similarity: # We know from __init__() which one we use, infer the mode from there instead of passing more arguments to forward
        print("TODO")
        # Prepare data accordingly
        # Call cosine sim model
      elif use_cross_encoder:
        print("TODO")

        # Call cross encoder model
    else:
      # Feed into document model with appropriate query types and labels
      if use_cosine_similarity:
        print("TODO")
        # Call cosine sim model
      elif use_cross_encoder:
        print("TODO")
        # Call cross encoder model