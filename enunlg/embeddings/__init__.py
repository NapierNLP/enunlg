import torch.nn
from omegaconf import DictConfig

import enunlg.embeddings.glove

__all__ = ["glove", "binary"]

CONFIG_ERR_MESSAGE = "Expected config to specify the type of embedding to use (can be 'torch' or 'glove')."


def embedding_model_from_config(config: DictConfig):
    """
    Instantiates a (sub)class of torch.nn.Embedding based on `config`

    Config can include any valid kwargs for torch.nn.Embedding if type is torch:
        type: 'torch'
        num_embeddings: null
        embedding_dim: 50
        padding_idx: null
        max_norm: null
        norm_type: 2.0
        scale_grad_by_freq: False
        sparse: False
        _weight: null
        _freeze: False
        device: null
        dtype: null

    If type is glove, then the config needs to include a filename for where the vectors should be loaded from,
    along with any valid kwargs for torch.nn.Embedding.from_pretrained():
        type: 'glove'
        vectors_file: null
        freeze: True
        padding_idx: null
        max_norm: null
        norm_type: 2.0
        scale_grad_by_freq: False
        sparse: False
    """
    kwargs = config.copy()
    if 'type' in kwargs:
        del kwargs['type']
    else:
        raise ValueError(CONFIG_ERR_MESSAGE)

    if config.type == "torch":
        return torch.nn.Embedding(**kwargs)  # type: ignore[misc]
    elif config.type == 'glove':
        filepath = config.vectors_file
        del kwargs['vectors_file']
        return enunlg.embeddings.glove.GloVeEmbeddings.from_word_embedding_txt(filepath, **kwargs)  # type: ignore[misc]
    else:
        raise ValueError(CONFIG_ERR_MESSAGE)