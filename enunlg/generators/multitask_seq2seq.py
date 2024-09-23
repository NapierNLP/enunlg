from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional

import logging
import tarfile
import tempfile

import omegaconf
import torch

import enunlg.data_management
import enunlg.encdec.multitask_seq2seq
import enunlg.normalisation.tokenisation
import enunlg.vocabulary

logger = logging.getLogger(__name__)


class MultitaskSeq2SeqGenerator(object):
    STATE_ATTRIBUTES = ('layers', 'vocabularies', 'max_length_any_layer', 'corpus_metadata', 'model')

    def __init__(self, corpus: enunlg.data_management.pipelinecorpus.TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.annotation_layers
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: enunlg.vocabulary.TokenVocabulary(list(corpus.items_by_layer(layer))) for layer in self.layers}  # type: ignore[misc]
        # Store some basic information about the corpus
        self.max_length_any_layer = corpus.max_layer_length
        self.corpus_metadata = corpus.metadata
        self.model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn(self.layers, [self.vocabularies[layer].size for layer in self.vocabularies], model_config)
        self.metadata = {}

    @property
    def input_layer_name(self) -> str:
        return self.layers[0]

    @property
    def output_layer_name(self):
        return self.layers[-1]

    @property
    def decoder_target_layer_names(self):
        return self.layers[1:]

    @property
    def model_config(self):
        return self.model.config

    def predict(self, mr):
        return self.model.generate(mr)

    def _save_classname_to_dir(self, directory_path):
        with (Path(directory_path) / "__class__.__name__").open('w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True):
        Path(filepath).mkdir()
        self._save_classname_to_dir(filepath)
        state = {}
        for attribute in self.STATE_ATTRIBUTES:
            curr_obj = getattr(self, attribute)
            save_method = getattr(curr_obj, 'save', None)
            # print(curr_obj)
            # print(save_method)
            attr_path = Path(filepath) / attribute
            if attribute == "vocabularies":
                # Special handling for the vocabularies
                state[attribute] = {}
                attr_path.mkdir()
                for key in curr_obj:
                    state[attribute][key] = f"./{attribute}/{key}"
                    curr_obj[key].save(attr_path / key, tgz=False)
            elif save_method is None:
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
    def load(cls, filepath):
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
                new_generator.metadata['loaded_model'] = filepath
                return new_generator

    def prep_embeddings(self, corpus, max_length: Optional[int] = None):
        layer_names = list(self.vocabularies.keys())
        input_layer_name = layer_names[0]
        if max_length is None:
            max_length = corpus.max_layer_length
        input_embeddings = [torch.tensor(self.vocabularies[input_layer_name].get_ints_with_left_padding(item, max_length),
                                         dtype=torch.long) for item in corpus.items_by_layer(input_layer_name)]
        output_embeddings = {
            layer_name: [torch.tensor(self.vocabularies[layer_name].get_ints(item), dtype=torch.long)
                         for item in corpus.items_by_layer(layer_name)]
            for layer_name in layer_names[1:]
        }
        return input_embeddings, output_embeddings

    def generate_output_corpus(self, slot_value_corpus, text_corpus, include_slot_value_corpus=False):
        max_input_length = self.model.max_input_length - 2

        # # drop entries that are too long
        # indices_to_drop = []
        # for idx, entry in enumerate(text_corpus):
        #     if len(entry['raw_input']) > max_input_length:
        #         indices_to_drop.append(idx)
        #         break
        # logger.info(f"Dropping {len(indices_to_drop)} entries for having too long an input rep.")
        # for idx in reversed(indices_to_drop):
        #     slot_value_corpus.pop(idx)
        #     text_corpus.pop(idx)

        # Prepare the embeddings
        test_input, test_output = self.prep_embeddings(text_corpus, max_input_length)
        test_ref = test_output['raw_output']
        logger.info(f"Num. input embeddings: {len(test_input)}")
        logger.info(f"Num. input refs: {len(test_ref)}")
        outputs = [self.model.generate_beam(embedding)[0][1] for embedding in test_input]
        logger.info("Done with generation.")

        # Convert to text
        best_outputs = [self.vocabularies['raw_output'].pretty_string(output).replace("<GO> ", "").replace(" <STOP>", "")
                        for output in outputs]
        # TODO move Enriched{E2E,WebNLG}-specific formatting out of this function
        ref_outputs = [self.vocabularies['raw_output'].pretty_string(output).replace(" @ ", " ").replace("<GO> ", "").replace(" <STOP>", "")
                       for output in test_ref]
        # Relexicalise
        relexed_best = []
        relexed_refs = []
        for sv_entry, best, ref in zip(slot_value_corpus, best_outputs, ref_outputs):
            curr_best = best
            curr_ref = ref
            for slot in sv_entry.raw_input.relex_dict:
                curr_best = curr_best.replace(slot, sv_entry.raw_input.relex_dict[slot].replace("_", " "))
                curr_ref = curr_ref.replace(slot, sv_entry.raw_input.relex_dict[slot].replace("_", " "))
            relexed_best.append(enunlg.normalisation.tokenisation.INLG2024Tokenizer.detokenise(curr_best))
            relexed_refs.append(enunlg.normalisation.tokenisation.INLG2024Tokenizer.detokenise(curr_ref))
        for best, ref in zip(relexed_best[:10], relexed_refs[:10]):
            logger.info(best)
            logger.info(ref)

        output_corpus = deepcopy(text_corpus)
        output_corpus.metadata['generator_metadata'] = self.metadata
        output_corpus.annotation_layers.append('best_output_indices')
        output_corpus.annotation_layers.append('best_output')
        output_corpus.annotation_layers.append('best_output_relexed')
        output_corpus.annotation_layers.append('ref_relexed')
        for indices, best, best_relexed, ref_relexed, entry in zip(outputs, best_outputs, relexed_best, relexed_refs, output_corpus):
            entry['best_output_indices'] = str([int(x) for x in indices])
            entry['best_output'] = best
            entry['best_output_relexed'] = best_relexed
            entry['ref_relexed'] = ref_relexed
            # entry.annotation_layers.append('best_output')
            # entry.annotation_layers.append('relexed_reference')
        if include_slot_value_corpus:
            return output_corpus, slot_value_corpus
        else:
            return output_corpus



class SingleVocabMultitaskSeq2SeqGenerator(MultitaskSeq2SeqGenerator):
    STATE_ATTRIBUTES = ('layers', 'vocabularies', 'max_length_any_layer', 'corpus_metadata', 'model')

    def __init__(self, corpus: enunlg.data_management.pipelinecorpus.TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        super().__init__(corpus, model_config)
        all_layers_as_text = []
        for layer_name in self.layers:
            all_layers_as_text.extend(list(corpus.items_by_layer(layer_name)))
        self.vocabulary = enunlg.vocabulary.TokenVocabulary(all_layers_as_text)
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: self.vocabulary for layer in self.layers}  # type: ignore[misc]
        # Store some basic information about the corpus
        self.max_length_any_layer = corpus.max_layer_length
        self.corpus_metadata = corpus.metadata
        self.embedding = torch.nn.Embedding(self.vocabulary.size, 50)
        self.model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn(self.layers, [self.vocabulary.size for layer in self.vocabularies], model_config)
        self.model.encoder.add_module('embedding', self.embedding)
        for layer in self.model.task_decoders:
            self.model.task_decoders[layer].add_module('output_embeddings', self.embedding)
