import torch
from datasets import CLS, SEP


def pad_input_tokens(batch):
  """
  Receives a batch of documents and pads them to the longest document in the 
  batch so that each batch has documents with the same length but different
  batches can have documents with different length.
  NOTE: This does not affect the padding of sentences *within* documents! The
        individual sentences of a document get padded to the length of the
        document's longest sentence by the tokenizer.
  Args:
    batch: A list of length [batch_size] specified within the DataLoader that
      contains documents and their model-specific information as a Dictionary
      which itself contains torch.Tensors of size (doc_length, longest_seq_in_doc)
      where longest_seq_in_doc is computed by the Tokenizer during retrieval 
      from the Dataset.
  Returns:
    (longest_document, longest_sentence, batch): 
      A triple where the first entry is the longest doc in the batch before padding. 
      It's passed to the model as a utility because the model uses that size for 
      intermediary padding.
      The second entry is the length of longest sentence as a list with length
      [batch_size] that holds the length of the longest sentence of each document.
      That number was used by the Tokenizer to pad sentences document-wise.
      The third entry is the appropriately padded batch ready to be passed to
      the model.
  """
  longest_document = max(len(doc['input_ids']) for doc in batch)
  doc_length_before_padding = []
  
  for doc in batch:
    doc_length_before_padding.append(len(doc['input_ids']))
    if len(doc['input_ids']) < longest_document:
      doc['input_ids'] = torch.Tensor.long(torch.cat((doc['input_ids'], torch.zeros(longest_document - len(doc['input_ids']), doc['input_ids'].size()[1])), dim=0))
      doc['token_type_ids'] = torch.Tensor.long(torch.cat((doc['token_type_ids'], torch.zeros(longest_document - len(doc['token_type_ids']), doc['token_type_ids'].size()[1])), dim=0))
      doc['attention_mask'] = torch.Tensor.long(torch.cat((doc['attention_mask'], torch.zeros(longest_document - len(doc['attention_mask']), doc['attention_mask'].size()[1])), dim=0))
      doc['special_tokens_mask'] = torch.Tensor.long(torch.cat((doc['special_tokens_mask'], torch.zeros(longest_document - len(doc['special_tokens_mask']), doc['special_tokens_mask'].size()[1])), dim=0))

  return (longest_document, doc_length_before_padding, batch)




def pad_input_embeds(batch):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  CLS_128 = torch.unsqueeze(CLS, dim=0)

  longest_document = max(len(doc.tensors[0]) for doc in batch)
  if longest_document > 512:
    longest_document = 512

  input_embeds_list = []
  attention_mask_list = []
  special_tokens_mask_list = []

  for doc in batch:
    if len(doc.tensors[0]) < longest_document:
      embedding = doc.tensors[0].to(device)
      padded_embeds = (torch.cuda.FloatTensor(torch.cat((CLS_128.to(device), embedding, torch.randn(longest_document - len(embedding), embedding.size()[1], device=device)), dim=0)))
      input_embeds_list.append(padded_embeds)

      attention = doc.tensors[1].to(device)
      padded_attention = (torch.Tensor.long(torch.cat((torch.ones(1, device=device), attention, torch.zeros(longest_document - len(attention), device=device)), dim=0)))
      attention_mask_list.append(padded_attention)

      masks = doc.tensors[2].to(device)
      padded_special_tokens = (torch.Tensor.long(torch.cat((torch.ones(1, device=device), masks, torch.ones(longest_document - len(masks), device=device)), dim=0)))
      special_tokens_mask_list.append(padded_special_tokens)

    elif len(doc.tensors[0]) == longest_document:
      embedding = doc.tensors[0].to(device)
      embedding = torch.cuda.FloatTensor(torch.cat((CLS_128.to(device), embedding), dim=0))
      input_embeds_list.append(embedding)

      attention = doc.tensors[1].to(device)
      attention = torch.cuda.FloatTensor(torch.cat((torch.ones(1, device=device), attention), dim=0))
      attention_mask_list.append(attention)

      masks = doc.tensors[2].to(device)
      masks = torch.cuda.FloatTensor(torch.cat((torch.ones(1, device=device), masks), dim=0))
      special_tokens_mask_list.append(masks)

  input_embeds = torch.stack(input_embeds_list)
  attention_mask = torch.stack(attention_mask_list)
  special_tokens_mask = torch.stack(special_tokens_mask_list)

  return input_embeds, attention_mask, special_tokens_mask




def prepare_msmarco_for_cross_encoder(batch):

  hidden_size = len(batch[0].tensors[0][0])

  CLS_128 = torch.unsqueeze(CLS, dim=0)
  SEP_128 = torch.unsqueeze(SEP, dim=0)
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  longest_document = max(len(pair.tensors[1][0]) for pair in batch)
  if longest_document > 509:
    longest_document = 509

  just_padded_queries =  []
  just_padded_documents = []
  pairs_embeddings_negative_list = []

  query_document_embeddings = []
  query_document_attention_mask = []
  query_document_token_types = []
  #query_document_special_tokens = []


  for i, query_doc_pair in enumerate(batch):
    if len(query_doc_pair.tensors[1][0]) <= longest_document:
      CLS_128 = CLS_128.to(device)
      query = query_doc_pair.tensors[0].to(device)
      SEP_128 = SEP_128.to(device)
      document = query_doc_pair.tensors[1][0].to(device)
      doc_length = len(document)
      padding = torch.randn(longest_document - doc_length, hidden_size, device=device)

      full_embedding = torch.cat((CLS_128, query, SEP_128, document, padding), 0)
      query_document_embeddings.append(full_embedding)
      just_padded_queries.append(torch.cat((CLS_128, query, SEP_128), 0))
      just_padded_documents.append(torch.cat((document, padding), 0))

      attention = torch.cat((torch.ones(3 + doc_length, dtype=torch.long), torch.zeros(longest_document - doc_length, dtype=torch.long))).to(device)
      query_document_attention_mask.append(attention)

      token_types = torch.cat((torch.zeros(3, dtype=torch.long), torch.ones(longest_document, dtype=torch.long))).to(device)
      query_document_token_types.append(token_types)

      #special_tokens = torch.cat((torch.Tensor([1,0,1]), torch.zeros(doc_length), torch.ones(longest_document - doc_length)))
      #query_document_special_tokens.append(special_tokens)

  pairs_embeddings_positive = torch.stack(query_document_embeddings)
  pairs_attention_positive = torch.stack(query_document_attention_mask)
  pairs_token_types_positive = torch.stack(query_document_token_types)
  #pairs_special_tokens_positive = torch.stack(query_document_special_tokens)

  queries_unshuffled = torch.stack(just_padded_queries)
  docs_unshuffled = torch.stack(just_padded_documents)
  
  indexes = torch.randperm(docs_unshuffled.shape[0])
  docs_shuffled = docs_unshuffled[indexes]
  for i, random_doc in enumerate(docs_shuffled):
    negative_pair = torch.cat((queries_unshuffled[i], random_doc))
    pairs_embeddings_negative_list.append(negative_pair)

  pairs_embeddings_negative = torch.stack(pairs_embeddings_negative_list)
  pairs_attention_negative = pairs_attention_positive[indexes]
  pairs_token_types_negative = pairs_token_types_positive[indexes]
  #pairs_special_tokens_negative = pairs_special_tokens_positive[indexes]

  embeddings = torch.cat((pairs_embeddings_positive, pairs_embeddings_negative))
  attention_mask = torch.cuda.LongTensor(torch.cat((pairs_attention_positive, pairs_attention_negative)))
  token_types = torch.cuda.LongTensor(torch.cat((pairs_token_types_positive, pairs_token_types_negative)))
  #special_tokens = torch.cuda.LongTensor(torch.cat((pairs_special_tokens_positive, pairs_special_tokens_negative)))

  return embeddings, attention_mask, token_types#, special_tokens




def prepare_msmarco_topx_for_cross_encoder(batch):
  """
  Collate function that can load datasets with x number of initial retrievals
  per query. The retrievals are stored in a list of lists where the first
  dimension marks distinct query IDs and the corresponding retrievals are stored
  in the second dimension.
  Can be used in DataLoaders for both "hard-mode" training with in-query negatives
  or for evaluation.
  Note that for evaluation the batch size should be set to 1 in the DataLoader.

  Args:
    batch: list of lists consisting of batch_size number of query IDs and their
      corresponding retrievals
  """

  hidden_size = len(batch[0][0].tensors[1][0])

  CLS_128 = torch.unsqueeze(CLS, dim=0)
  SEP_128 = torch.unsqueeze(SEP, dim=0)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  longest_document = 0
  for query in batch:
    max_per_query = max(len(retrieval.tensors[2][0]) for retrieval in query)
    if max_per_query > longest_document:
      longest_document = max_per_query

  labels = []
  query_document_embeddings = []
  query_document_attention_mask = []
  query_document_token_types = []
  
  for query in batch:
    labels_per_query = []
    for retrieval in query:
      if len(retrieval.tensors[2][0]) <= longest_document:

        CLS_128 = CLS_128.to(device)
        query = retrieval.tensors[1].to(device)
        SEP_128 = SEP_128.to(device)
        document = retrieval.tensors[2][0].to(device)
        doc_length = len(document)
        padding = torch.randn(longest_document - doc_length, hidden_size, device=device)

        full_embedding = torch.cat((CLS_128, query, SEP_128, document, padding), 0)
        query_document_embeddings.append(full_embedding)
        
        attention = torch.cat((torch.ones(3 + doc_length, dtype=torch.long), torch.zeros(longest_document - doc_length, dtype=torch.long))).to(device)
        query_document_attention_mask.append(attention)

        token_types = torch.cat((torch.zeros(3, dtype=torch.long), torch.ones(longest_document, dtype=torch.long))).to(device)
        query_document_token_types.append(token_types)

        labels_per_query.append(retrieval.tensors[0][0][0].item())

    labels.append(torch.LongTensor(labels_per_query))

  labels = torch.stack(labels).to(device)

  embeddings = torch.stack(query_document_embeddings)
  pairs_attention_positive = torch.stack(query_document_attention_mask)
  pairs_token_types_positive = torch.stack(query_document_token_types)
  #pairs_special_tokens_positive = torch.stack(query_document_special_tokens)

  if torch.cuda.is_available():
    attention_mask = torch.cuda.LongTensor(pairs_attention_positive)
    token_types = torch.cuda.LongTensor(pairs_token_types_positive)
  else:
    attention_mask = torch.LongTensor(pairs_attention_positive)
    token_types = torch.LongTensor(pairs_token_types_positive)
  #special_tokens = torch.cuda.LongTensor(torch.cat((pairs_special_tokens_positive, pairs_special_tokens_negative)))

  return embeddings, attention_mask, token_types, labels#, query_ids,




def prepare_msmarco_for_cosine_similarity(batch):

  hidden_size = len(batch[0].tensors[0][0])
  CLS_128 = torch.unsqueeze(CLS, dim=0)
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  longest_document = max(len(pair.tensors[1][0]) for pair in batch)
  if longest_document > 509:
    longest_document = 509

  query_embeddings_list = []
  docs_padded_list = []
  attention_list = []

  for i, query_doc_pair in enumerate(batch):
    if len(query_doc_pair.tensors[1][0]) <= longest_document:

      query = query_doc_pair.tensors[0][0].to(device)
      query_embeddings_list.append(query)

      CLS_128 = CLS_128.to(device)
      document = query_doc_pair.tensors[1][0].to(device)
      doc_length = len(document)
      padding = torch.randn(longest_document - doc_length, hidden_size, device=device)

      doc_padded = torch.cat((CLS_128, document, padding), 0)
      docs_padded_list.append(doc_padded)

      attention = torch.cat((torch.ones(1 + doc_length, dtype=torch.long), torch.zeros(longest_document - doc_length, dtype=torch.long))).to(device)
      attention_list.append(attention)

  query_embeddings = torch.stack(query_embeddings_list)
  docs_embeddings = torch.stack(docs_padded_list)
  attention_mask = torch.stack(attention_list)

  return query_embeddings, docs_embeddings, attention_mask




def prepare_msmarco_topx_for_cosine_similarity(batch):

  hidden_size = len(batch[0][0].tensors[1][0])
  CLS_128 = torch.unsqueeze(CLS, dim=0)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  longest_document = 0
  for query in batch:
    max_per_query = max(len(retrieval.tensors[2][0]) for retrieval in query)
    if max_per_query > longest_document:
      longest_document = max_per_query

  labels = []

  query_embeddings_list = []
  docs_padded_list = []
  attention_list = []

  for query in batch:
    labels_per_query = []
    for retrieval in query:
      if len(retrieval.tensors[2][0]) <= longest_document:
        query = retrieval.tensors[1].to(device)
        query_embeddings_list.append(query)

        CLS_128 = CLS_128.to(device)
        document = retrieval.tensors[2][0].to(device)
        doc_length = len(document)
        padding = torch.randn(longest_document - doc_length, hidden_size, device=device)

        doc_padded = torch.cat((CLS_128, document, padding), 0)
        docs_padded_list.append(doc_padded)

        attention = torch.cat((torch.ones(1 + doc_length, dtype=torch.long), torch.zeros(longest_document - doc_length, dtype=torch.long))).to(device)
        attention_list.append(attention)

        labels_per_query.append(retrieval.tensors[0][0][0].item())

    labels.append(torch.LongTensor(labels_per_query))

  labels = torch.stack(labels).to(device)


  query_embeddings = torch.stack(query_embeddings_list)
  docs_embeddings = torch.stack(docs_padded_list)
  attention_mask = torch.stack(attention_list)

  return query_embeddings, docs_embeddings, attention_mask, labels