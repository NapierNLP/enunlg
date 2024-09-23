import tarfile
import tempfile
from pathlib import Path

import numpy as np
import omegaconf

from enunlg.nlu import binary_mr_classifier

import enunlg
import enunlg.embeddings.binary
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.util
import enunlg.vocabulary


class FullBinaryMRClassifier(object):
    STATE_ATTRIBUTES = ('text_vocab', 'binary_mr_vocab', 'model')

    def __init__(self, text_vocab: enunlg.vocabulary.TokenVocabulary,
                 binary_mr_vocab: enunlg.embeddings.binary.DialogueActEmbeddings,
                 model_config: omegaconf.DictConfig):
        """
        Create a classifier and it's associated files
        """
        self.text_vocab = text_vocab
        self.binary_mr_vocab = binary_mr_vocab
        # Store some basic information about the corpus
        self.model = binary_mr_classifier.TGenSemClassifier(self.text_vocab.size,
                                                            self.binary_mr_vocab.dimensionality,
                                                            model_config)
        self.metadata = {}

    def prepare_input(self, corpus):
        return [das.MultivaluedDA.from_slot_value_list('inform', mr.items())
                for mr in corpus.items_by_layer('raw_input')]

    @property
    def model_config(self):
        return self.model.config

    def predict(self, text_ints):
        return self.model.predict(text_ints).squeeze(0).squeeze(0).tolist()

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
            if save_method is None:
                state[attribute] = curr_obj
            else:
                state[attribute] = f"./{attribute}"
                curr_obj.save(f"{filepath}/{attribute}", tgz=False)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=Path(filepath).name)

    @classmethod
    def load(cls, filepath):
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r') as classifier_file:
                tmp_dir = tempfile.mkdtemp()
                tarfile_member_names = classifier_file.getmembers()
                classifier_file.extractall(tmp_dir)
                # print([x.name for x in tarfile_member_names])
                root_name = Path(tarfile_member_names[0].name).parts[0]
                root_dir = Path(tmp_dir) / root_name
                # print(root_dir)
                return cls.load_from_dir(root_dir, source_file_name=str(filepath))
        elif Path(filepath).is_dir():
            return cls.load_from_dir(filepath)
        else:
            message = f"filepath must be the location of a tarfile or a directory, but this is neither:\n{filepath}"
            raise ValueError(message)

    @classmethod
    def load_from_dir(cls, filepath, source_file_name=None):
        with (Path(filepath) / "__class__.__name__").open('r') as class_name_file:
            class_name = class_name_file.read().strip()
            assert class_name == cls.__name__, f"{class_name} != {cls.__name__}"
        model = enunlg.nlu.binary_mr_classifier.TGenSemClassifier.load_from_dir(Path(filepath) / 'model')
        text_vocab = enunlg.vocabulary.TokenVocabulary.load_from_dir(Path(filepath) / 'text_vocab')
        binary_mr_vocab = enunlg.embeddings.binary.DialogueActEmbeddings.load_from_dir(Path(filepath) / 'binary_mr_vocab')
        classifier = cls(text_vocab, binary_mr_vocab, model.config)
        classifier.model = model
        if source_file_name is None:
            source_file_name = filepath
        classifier.metadata['filepath'] = source_file_name
        return classifier

    def evaluate(self, test_pairs):
        error = 0
        for i, o in test_pairs:
            prediction = self.predict(i)
            target_bitvector = np.round(o.tolist())
            output_bitvector = np.round(prediction)
            # print(prediction)
            # print(target_bitvector)
            # print(output_bitvector)
            current = enunlg.util.hamming_error(target_bitvector, output_bitvector)
            # print(current)
            error += current
        return error / len(test_pairs)