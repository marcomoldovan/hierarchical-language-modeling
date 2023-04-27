import torch


def load_weights(model, weights_path, device):
  pretrained_dict = torch.load(weights_path, map_location=device)['model_state_dict']
  model_dict = model.state_dict()

  # 1. filter out unnecessary keys
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  # 2. overwrite entries in the existing state dict
  model_dict.update(pretrained_dict) 
  # 3. load the new state dict
  model.load_state_dict(pretrained_dict, strict=False)

  return model


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)