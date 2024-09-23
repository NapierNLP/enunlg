import omegaconf

import enunlg.trainer.seq2seq


class TGenTrainer(enunlg.trainer.seq2seq.Seq2SeqAttnTrainer):
    def __init__(self,
                 model: "enunlg.encdec.tgen.TGenEncDec",
                 training_config=None,
                 input_vocab=None,
                 output_vocab=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                    "record_interval": 1000,
                                                    "shuffle": True,
                                                    "batch_size": 1,
                                                    "optimizer": "adam",
                                                    "learning_rate": 0.0005,
                                                    "learning_rate_decay": 0.5  # TGen used 0.0
                                                   })
        super().__init__(model, training_config, input_vocab=input_vocab, output_vocab=output_vocab)
