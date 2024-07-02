import torch
from torch import nn
from typing import Optional

from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from tunip.config import Config


class TargetValueOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


class TargetValueModel(nn.Module):
    def __init__(self, config: Config):
        super(TargetValueModel, self).__init__()
        self.embed_dim = config.d_model

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Linear(2 * self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, 1)

    
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        decoder_inputs_embeds: torch.FloatTensor,
        labels: torch.FloatTensor,
        return_dict: Optional[bool] = None
    ):
        hidden_states = self.fc1(torch.concat([inputs_embeds, decoder_inputs_embeds], dim=1))
        outputs = self.fc2(hidden_states)

        loss = None
        if labels is not None:
            criterion = nn.MSELoss()
            loss = torch.sqrt(criterion(outputs, labels))
        
        if not return_dict:
            return ((loss,)+ tuple(outputs)) if loss is not None else outputs

        return TargetValueOutput(
            loss=loss,
            scores=outputs
        ) 
