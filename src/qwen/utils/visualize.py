import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_embeddings(sentences, batch_size=8):

    embeddings = []

    for i in range(0, len(sentences), batch_size):

        batch = sentences[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model_encoder.device)

        with torch.no_grad():
            outputs = model_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        print(len(hidden_states))

        layer_outputs = []

        for layer in range(0, len(hidden_states), 1):  # layer chẵn

            hidden = hidden_states[layer]

            mask = inputs["attention_mask"].unsqueeze(-1)

            pooled = (hidden * mask).sum(1) / mask.sum(1)

            layer_outputs.append(pooled.cpu())

        embeddings.append(torch.stack(layer_outputs))

    embeddings = torch.cat(embeddings, dim=1)

    return embeddings