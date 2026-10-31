# -*- coding:utf-8 -*-
from transformers import RobertaModel, BertPreTrainedModel
import torch.nn as nn

class RobertaForSequenceClassification_Modified(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        is_feat=False,
    ):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:,0,:]
        logits = self.classifier(cls_embedding)

        if is_feat:
            return cls_embedding, logits
        return logits
