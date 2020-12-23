import torch
import torch.nn as nn
import torch.nn.functional as f

import transformers
from transformers import BertConfig, BertModel, BertForMaskedLM
from transformers import configuration_utils
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward
from transformers.configuration_utils import PretrainedConfig


# TODO add models from the notebook in here

