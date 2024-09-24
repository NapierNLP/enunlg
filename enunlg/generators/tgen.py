from pathlib import Path
from typing import List, Tuple

import tarfile
import tempfile

import omegaconf
import torch

import enunlg.data_management
import enunlg.encdec.multitask_seq2seq
import enunlg.vocabulary


class TGenGenerator(object):
    STATE_ATTRIBUTES = ('input_vocab', 'output_vocab', 'corpus_metadata', 'model')

    def __init__(self, corpus: enunlg.data_management.e2e_challenge.E2ECorpus, model_config: omegaconf.DictConfig):
        """
        Create a TGen model based on `corpus`.
        """
        self.input_vocab = enunlg.vocabulary.IntegralInformVocabulary([mr for mr, _ in corpus])
        self.output_vocab = enunlg.vocabulary.TokenVocabulary([text.strip().split() for _, text in corpus])

        # Store some basic information about the corpus
        self.corpus_metadata = corpus.metadata
        self.model = enunlg.encdec.tgen.TGenEncDec(self.input_vocab.size, self.output_vocab.size, model_config=model_config)

    @property
    def model_config(self) -> omegaconf.DictConfig:
        return self.model.config

    def predict(self, mr):
        return self.model.generate(mr)

    def _save_classname_to_dir(self, directory_path) -> None:
        with (Path(directory_path) / "__class__.__name__").open('w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True) -> None:
        Path(filepath).mkdir()
        self._save_classname_to_dir(filepath)
        state = {}
        for attribute in self.STATE_ATTRIBUTES:
            curr_obj = getattr(self, attribute)
            save_method = getattr(curr_obj, 'save', None)
            attr_path = Path(filepath) / attribute
            if save_method is None:
                state[attribute] = curr_obj
            else:
                state[attribute] = f"./{attribute}"
                curr_obj.save(attr_path, tgz=False)
        with (Path(filepath) / "_save_state.yaml").open('w') as state_file:
            state = omegaconf.OmegaConf.create(state)
            omegaconf.OmegaConf.save(state, state_file)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=Path(filepath).parent)

    @classmethod
    def load(cls, filepath) -> "TGenGenerator":
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r') as generator_tarball:
                tmp_dir = tempfile.mkdtemp()
                tarfile_members = generator_tarball.getmembers()
                generator_tarball.extractall(tmp_dir)
                root_name = Path(tarfile_members[0].name).parts[0]
                root_dir = Path(tmp_dir) / root_name
                with (root_dir / "__class__.__name__").open('r') as class_name_file:
                    class_name = class_name_file.read().strip()
                    assert class_name == cls.__name__
                model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn.load_from_dir(root_dir / 'model')
                dummy_pipeline_item = enunlg.data_management.pipelinecorpus.PipelineItem({layer_name: "" for layer_name in model.layer_names})
                dummy_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus([dummy_pipeline_item])
                dummy_corpus.pop()
                new_generator = cls(dummy_corpus, model.config)
                new_generator.model = model
                state_dict = omegaconf.OmegaConf.load(root_dir / "_save_state.yaml")
                vocabs = {}
                for vocab in state_dict.vocabularies:
                    vocabs[vocab] = enunlg.vocabulary.TokenVocabulary.load_from_dir(root_dir / 'vocabularies' / vocab)
                new_generator.vocabularies = vocabs
                return new_generator

    def prep_embeddings(self, corpus, max_input_length_in_kv_pairs: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        input_embeddings = [torch.tensor(self.input_vocab.get_ints_with_padding(mr, max_input_length_in_kv_pairs), dtype=torch.long)
                            for mr, _ in corpus]
        output_embeddings = [torch.tensor(self.output_vocab.get_ints(text.strip().split()), dtype=torch.long)
                             for _, text in corpus]
        return input_embeddings, output_embeddings
