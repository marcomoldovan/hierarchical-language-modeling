import pickle
import torch
import transformers
from nltk import sent_tokenize
from torch.utils.data import Dataset



CLS = torch.Tensor([-0.9981,  1.1354, -3.0456, -1.6170, -0.8531,  0.1014,  0.2313,  0.6844,
                    -0.8162,  1.8423,  0.6286, -0.2557, -3.2614,  0.4459,  1.6963, -1.2250,
                    -0.4966, -0.2596, -2.1223,  0.8322, -0.1752, -0.4665,  3.0415,  0.8801,
                     0.3117,  0.1467,  0.4205,  0.5684,  0.0836, -0.0689, -1.4866, -1.3490,
                    -1.1071,  0.3617,  1.7387, -1.0797,  0.3693, -0.8626, -1.4490, -1.6486,
                     0.9362, -1.3971,  0.8101, -1.2994, -0.2535, -1.9793, -0.8475, -1.2452,
                     0.1394, -0.0303, -0.7455,  1.4981, -0.0614, -0.3954,  1.4995, -1.3618,
                     0.2045,  0.9455, -1.0810,  1.8007,  1.3504, -0.3771, -1.0584,  0.0370,
                    -0.5903,  0.4348,  0.4075, -0.0932,  1.4073, -0.6776, -0.2490,  0.5176,
                    -0.9007,  0.4315,  0.3015,  0.0353,  0.5067,  0.7153,  2.1073,  0.3022,
                    -0.4577,  0.5652, -0.2365,  0.1794, -0.4954, -0.5183, -0.2060, -0.7410,
                     2.8978, -2.0026,  1.5776,  0.1521, -1.3113,  1.8437,  1.0323,  0.9030,
                     0.4231, -0.5792, -0.1886,  0.4181, -1.1298,  0.1378,  0.1590, -0.7804,
                     1.0177, -1.4610, -0.3874,  0.7050,  0.8824, -1.5715,  0.6487,  0.6148,
                    -0.1988, -0.2373,  0.4794,  1.6276, -0.3890, -0.9612,  1.2258,  0.3845,
                    -0.3818,  1.0645,  1.4688, -0.8300, -1.6368, -0.9549, -0.9308, -1.2218])

SEP = torch.Tensor([-2.1891e+00,  1.2711e+00, -8.9792e-01, -1.5016e+00, -5.9451e-01,
                     4.3516e-01,  7.7865e-01,  1.9005e-01, -2.2061e-01,  1.1726e+00,
                     3.0554e-01, -7.1097e-01, -3.7538e+00,  6.0648e-02,  2.0107e+00,
                    -1.3236e+00, -8.7579e-01, -2.9496e-01, -1.8437e+00,  5.3592e-01,
                    -4.6524e-01, -4.3795e-01,  2.1429e+00,  1.1876e+00,  4.3196e-01,
                     2.0292e-01, -1.4503e-01,  6.9892e-01,  5.6384e-01,  4.6437e-01,
                    -1.9509e+00, -3.2635e-02,  3.2464e-01,  6.8599e-01,  1.3163e+00,
                    -1.7324e+00,  2.6175e-01, -7.4656e-01, -1.8667e+00, -1.5655e+00,
                     1.2984e+00, -7.0660e-01,  1.9371e-01, -9.8461e-01, -2.6006e-02,
                    -2.1590e+00, -8.7077e-01, -1.0568e+00, -1.8535e-02, -4.5890e-01,
                    -2.0912e-01,  1.7237e+00,  1.4897e-01, -3.0492e-02,  1.0122e+00,
                    -5.5029e-01,  1.6240e+00,  4.2045e-01, -1.2400e+00,  2.0293e+00,
                     4.3488e-01, -7.9258e-01, -1.3834e+00,  6.4267e-01, -6.0309e-01,
                    -7.2195e-04,  3.7046e-01,  5.0487e-01,  1.1453e+00,  8.6877e-02,
                    -1.6964e-02,  1.0068e+00, -1.0256e+00, -9.3181e-02,  3.8434e-01,
                     3.1522e-01, -1.7939e-01,  8.0017e-02,  9.9643e-01, -1.9349e-01,
                    -5.2971e-01, -8.9503e-01, -3.0495e-01,  2.0245e-01,  7.9598e-02,
                    -1.1440e+00, -4.3081e-02, -4.9467e-01,  2.0151e+00, -1.7306e+00,
                     1.0992e+00,  8.5625e-01, -1.9254e+00,  1.8504e+00,  1.3958e+00,
                     1.4260e+00,  3.1682e-01, -8.9479e-01,  6.8218e-01,  7.8508e-01,
                    -1.2306e+00,  4.1371e-01, -1.1960e-01, -1.0888e+00,  1.0746e+00,
                    -1.3507e+00, -2.8201e-01,  8.3991e-01,  3.8711e-01, -1.9244e+00,
                     1.9884e-01,  1.1749e+00,  5.3496e-01,  5.5470e-01,  1.0846e+00,
                     1.9696e+00, -8.7533e-01, -1.4858e+00,  6.4223e-01,  1.2655e+00,
                    -1.9945e-01,  8.3856e-01,  5.2890e-01, -1.1276e+00, -1.7738e+00,
                    -1.9558e+00, -1.1231e+00, -2.5221e-01])

print("CLS: ", CLS.size(), " SEP: ", SEP.size())


class PretrainingData(Dataset):
  
  def __init__(self, dataset):
    self.tokenizer = transformers.BertTokenizerFast.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    self.wiki_dump = dataset # dataset #load_dataset('wikipedia', '20200501.en', split='train')


  def __len__(self):
    return len(self.wiki_dump['text'])

  # TODO which of batch_encode_plus/encode_plus/prepare_for_model is better?
  def __getitem__(self, id):
    # Split document at dataset[id] into sentences
    doc_split = sent_tokenize(self.wiki_dump['text'][id])
    # Batch-encodes a whole document at dataset[id] at once
    doc_tokenized =  self.tokenizer.batch_encode_plus(doc_split,
                                                      padding='longest',
                                                      truncation='longest_first',
                                                      return_tensors='pt',
                                                      return_special_tokens_mask=True)
    return doc_tokenized
  
  
  
class PretrainingDataPreEncoded(Dataset):
  def __init__(self, filepath):
    with open(filepath, 'rb') as p:
      self.dataset = pickle.load(p)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]
  
  
  
  
class MSMarcoDocumentRankingDataset(Dataset):
  def __init__(self, filepath):
    with open(filepath, 'rb') as p:
      self.dataset = pickle.load(p)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]