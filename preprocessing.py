import time
import pickle
import torch
import transformers
from nltk import sent_tokenize


def pre_encode_wikipedia(model,
                         save_path,
                         wikipedia_small,
                         pretrained_sentence_model='google/bert_uncased_L-2_H-128_A-2'):
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tokenizer = transformers.BertTokenizerFast.from_pretrained(pretrained_sentence_model)
  document_data_list = []

  for iteration, document in enumerate(wikipedia_small['text']):
    sentence_embeds_per_doc = [torch.randn(128, device=device)]
    attention_mask_per_doc = [1]
    special_tokens_per_doc = [1]
    doc_split = sent_tokenize(document)
    doc_tokenized = tokenizer.batch_encode_plus(doc_split,
                                                padding='longest',
                                                truncation=True,
                                                max_length=512,
                                                return_tensors='pt')
    for key, value in doc_tokenized.items():
      doc_tokenized[key] = doc_tokenized[key].to(device)
    doc_encoded = model(**doc_tokenized)
    for sentence in doc_encoded['last_hidden_state']:
      sentence_embeds_per_doc.append(sentence[0])
      attention_mask_per_doc.append(1)
      special_tokens_per_doc.append(0)

    sentence_embeds = torch.stack(sentence_embeds_per_doc)
    if torch.cuda.is_available():
      attention_mask = torch.cuda.FloatTensor(attention_mask_per_doc)
      special_tokens_mask = torch.cuda.FloatTensor(special_tokens_per_doc)
    else:
      attention_mask = torch.FloatTensor(attention_mask_per_doc)
      special_tokens_mask = torch.FloatTensor(special_tokens_per_doc)
    
    sentence_embeds.to('cpu')
    attention_mask.to('cpu')
    special_tokens_mask.to('cpu')
    torch.cuda.empty_cache()
    document_data = torch.utils.data.TensorDataset(*[sentence_embeds, attention_mask, special_tokens_mask])
    document_data_list.append(document_data)
    print(f"Document at {iteration} encoded and appended.")
  
  with open(f'{save_path}{time.strftime("%Y%m%d-%H%M%S")}_16384.pkl', 'wb') as p:
    pickle.dump(document_data_list, p) 
    
  print(f"Batch saved in pickle file.")
  
  
  
  
