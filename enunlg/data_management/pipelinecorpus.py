from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, TypeVar, TYPE_CHECKING, Union

import logging
import random

from enunlg.meaning_representation.slot_value import SlotValueMR

import enunlg.data_management.iocorpus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import os


def string_to_python_identifier(text: str):
    return text.replace("-", "_").replace(" ", "_").strip()


class PipelineItem(object):
    def __init__(self, annotation_layers: Dict[str, Any]):
        """
        An entry in a pipeline dataset containing all the annotation_layers for a single example from the corpus.

        Each annotation layer with name `layer_name` for PipelineItem `x` can be accessed as x.layer_name or x['layer_name'].
        This is why layer names must be valid Python identifiers.

        :param annotation_layers: dict mapping layer names to the entry for that layer
        """
        self.annotation_layers = [string_to_python_identifier(layer_name) for layer_name in annotation_layers]
        self.metadata = {}
        for new_name, layer in zip(self.annotation_layers, annotation_layers):
            self.__setattr__(new_name, annotation_layers[layer])

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __delitem__(self, item):
        return self.__delattr__(item)

    def __repr__(self):
        attr_string = ", ".join([f"{layer}={str(self[layer])}" for layer in self.annotation_layers])
        return f"{self.__class__.__name__}({attr_string})"

    def print_layers(self):
        for layer in self.annotation_layers:
            if isinstance(self[layer], str):
                layer_content = self[layer]
            else:
                layer_content = " ".join(self[layer])
            print(f"{layer}|\t{layer_content}")

    def drop_layers(self, drop=None, keep=None) -> None:
        if drop is None and keep is None:
            raise ValueError("Must specify either which layers to keep or which layers to drop")
        if drop and keep:
            raise ValueError("Must only specify one of keep or drop, and specified both")
        if drop:
            # TODO implement this for completeness
            raise NotImplementedError("Haven't implemented this path yet...")
        if keep:
            for layer_name in self.annotation_layers:
                if layer_name not in keep:
                    del self[layer_name]
            self.annotation_layers = list(keep)


AnyPipelineItemSubclass = TypeVar("AnyPipelineItemSubclass", bound=PipelineItem)


class PipelineCorpus(enunlg.data_management.iocorpus.IOCorpus):
    STATE_ATTRIBUTES = ("metadata", "annotation_layers")

    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        """Each item in a PipelineCorpus is a single entry with annotations for each stage of the pipeline."""
        if seq is None:
            self.annotation_layers = []
        else:
            tmp = set(tuple(entry.annotation_layers) for entry in seq)
            # print(tmp)
            # for entry in seq:
            #     if len(entry.annotation_layers) == 1:
            #         print(entry)
            err_msg = f"Expected all items in seq to have the same layers, but found {tmp}"
            assert len(tmp) == 1, err_msg
            self.annotation_layers = list(tmp.pop())
        super(PipelineCorpus, self).__init__(seq)
        if metadata is None:
            self.metadata = {'name': None,
                             'splits': None,
                             'directory': None
                             }
        else:
            self.metadata = metadata

    # def __getstate__(self):
    #     state = {attribute: self.__getattribute__(attribute)
    #              for attribute in self.STATE_ATTRIBUTES}
    #     state['__class__'] = self.__class__.__name__
    #     state['_content'] = list(self)
    #     return state
    #
    # @classmethod
    # def __setstate__(cls, state: Dict[str, Any]):
    #     class_name = state["__class__"]
    #     assert class_name == cls.__name__
    #     new_generator = cls.__new__(cls)
    #     for attribute in cls.STATE_ATTRIBUTES:
    #         new_generator.__setattr__(attribute, state[attribute])
    #     new_generator.append(state['_content'])
    #     return new_generator

    @property
    def layer_pairs(self):
        """Layers are listed in order, so adjacent pairs of annotation layers form individual Pipeline subtasks."""
        return list(zip(self.annotation_layers, self.annotation_layers[1:]))

    def items_by_layer_pair(self, layer_pair: Tuple[str, str]):
        layer_from, layer_to = layer_pair
        for item in self:
            yield item[layer_from], item[layer_to]

    def items_by_layer(self, layer_name):
        for item in self:
            yield item[layer_name]

    def print_summary_stats(self):
        print(f"{self.metadata=}")
        print(", ".join(self.annotation_layers))
        print(f"num entries: {len(self)}")
        num_entries_per_layer = defaultdict(int)
        layer_lengths = defaultdict(list)
        layer_types = defaultdict(set)
        for layer in self.annotation_layers:
            for entry in self:
                if entry[layer] is not None:
                    layer_lengths[layer].append(len(entry[layer]))
                    if isinstance(entry[layer], SlotValueMR):
                        layer_types[layer].update(entry[layer])
                    elif isinstance(entry[layer], tuple):
                        if all(isinstance(x, SlotValueMR) for x in entry[layer]):
                            layer_types[layer].update(tuple(x for x in entry[layer]))
                    else:
                        layer_types[layer].update(entry[layer])
                    num_entries_per_layer[layer] += 1
                else:
                    print("None type found for the entry {}".format(entry))
        for layer in layer_lengths:
            print(f"{layer}:\t\t{sum(layer_lengths[layer])/num_entries_per_layer[layer]:.2f} [{min(layer_lengths[layer])},{max(layer_lengths[layer])}]")
            print(f"    with {len(layer_types[layer])} types across {sum(layer_lengths[layer])} tokens.")

    def print_sample(self, range_start=0, range_end=10, subsample=None):
        if subsample is None:
            for item in self[range_start:range_end]:
                item.print_layers()
                print("----")
        elif isinstance(subsample, int):
            for item in random.choices(self[range_start:range_end], k=subsample):
                item.print_layers()
                print("----")
        else:
            message = "`subsample` must be None or an integer"
            raise ValueError(message)

    def drop_layers(self, drop=None, keep=None):
        if drop is None and keep is None:
            raise ValueError("Must specify either which layers to keep or which layers to drop")
        if drop and keep:
            raise ValueError("Must only specify one of keep or drop, and specified both")
        if drop:
            # TODO implement this for completeness
            raise NotImplementedError("Haven't implemented this path yet...")
        if keep:
            if set(keep) == set(self.annotation_layers):
                pass
            else:
                for entry in self:
                    entry.drop_layers(drop, keep)
                self.annotation_layers = list(keep)


AnyPipelineCorpus = TypeVar("AnyPipelineCorpus", bound=PipelineCorpus)


class TextPipelineCorpus(PipelineCorpus):
    STATE_ATTRIBUTES = tuple(list(PipelineCorpus.STATE_ATTRIBUTES) + ["_max_layer_length", "_layer_lengths"])

    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        super(TextPipelineCorpus, self).__init__(seq, metadata)
        self._max_layer_length = -1
        self._layer_lengths = {layer_name: -1 for layer_name in self.annotation_layers}

    @classmethod
    def from_existing(cls, corpus: PipelineCorpus, mapping_functions):
        out_corpus = TextPipelineCorpus(deepcopy(corpus))
        out_corpus.metadata = corpus.metadata
        for item in out_corpus:
            for layer in item.annotation_layers:
                item[layer] = mapping_functions[layer](item[layer])
        return out_corpus

    @property
    def max_layer_length(self) -> int:
        if self._max_layer_length == -1:
            for item in self:
                for layer in item.annotation_layers:
                    if len(item[layer]) > self._max_layer_length:
                        logger.debug(f"New longest field, this time a {layer}")
                        logger.debug(item[layer])
                        self._max_layer_length = len(item[layer])
                    if len(item[layer]) > self._layer_lengths[layer]:
                        self._layer_lengths[layer] = len(item[layer])
        return self._max_layer_length

    def layer_length(self, layer_name: str) -> int:
        return self._layer_lengths[layer_name]

    def all_item_layer_iterator(self):
        for item in self:
            for layer in self.annotation_layers:
                yield item[layer]

    def save(self, filename: Union[str, bytes, "os.PathLike"]) -> None:
        with Path(filename).open('w') as out_file:
            self.write_to_iostream(out_file)
    
    def write_to_iostream(self, io_stream: TextIO) -> None:
        io_stream.write("# TextPipeline Corpus Save File\n")
        io_stream.write("# Format Version 0.3\n")
        io_stream.write("# \n")
        io_stream.write("# Metadata:\n")
        for key in self.metadata:
            if isinstance(self.metadata[key], dict):
                io_stream.write(f"#   {key}:\n")
                for inner_dict_key in self.metadata[key]:
                    io_stream.write(f"#     {inner_dict_key}: {self.metadata[key][inner_dict_key]}\n")
            else:
                io_stream.write(f"#   {key}: {self.metadata[key]}\n")
        io_stream.write("# \n")
        io_stream.write("# Annotation Layers:\n")
        for annotation_layer in self.annotation_layers:
            io_stream.write(f"#   {annotation_layer}\n")
        io_stream.write("\n")
        for entry in self:
            if 'eid' in entry.metadata and 'lid' in entry.metadata:
                io_stream.write(f"# {entry.metadata['eid']}-{entry.metadata['lid']}\n")
            for annotation_layer in self.annotation_layers:
                if isinstance(entry[annotation_layer], str):
                    layer_line = entry[annotation_layer]
                else:
                    layer_line = " ".join(entry[annotation_layer])
                io_stream.write(f"{layer_line}\n")
            io_stream.write("\n")

    @classmethod
    def load(cls, filename) -> "TextPipelineCorpus":
        with Path(filename).open('r') as input_file:
            in_header = True
            collect_header = []
            annotation_layer_names = []
            contents = []
            curr_entry_id = ""
            curr_entry_parts = []
            for line in input_file:
                if line.startswith("#"):
                    if in_header:
                        collect_header.append(line.strip("# \n"))
                    else:
                        if curr_entry_parts:
                            item = PipelineItem({x: y for x, y in zip(annotation_layer_names, curr_entry_parts)})
                            item.metadata['id'] = curr_entry_id
                            contents.append(item)
                        curr_entry_id = line.strip("# \n")
                        curr_entry_parts = []
                elif line.strip() == "":
                    if in_header:
                        in_header = False
                        corpus_metadata, annotation_layer_names = extract_data_from_header(collect_header)
                        # print(annotation_layer_names)
                else:
                    # print(line.strip())
                    curr_entry_parts.append(line.strip())
            try:
                retval = cls(contents)
            except AssertionError:
                raise

            retval.metadata = corpus_metadata
            retval.metadata['loaded_from'] = str(filename)
            return retval


def extract_data_from_header(header):
    in_metadata = False
    in_anno_layers = False
    in_a_subdict = False
    metadata = {}
    subdict_label = ""
    subdict = {}
    annotation_layers = []
    for line in header:
        if "TextPipeline Corpus Save File" in line:
            pass
        elif line.strip().startswith("Format Version"):
            pass
        elif line == "Metadata:":
            in_metadata = True
        elif line == "Annotation Layers:":
            in_metadata = False
            in_anno_layers = True
        elif line == "":
            in_metadata = False
            in_anno_layers = False
        else:
            if in_metadata:
                parts = line.split()
                if len(parts) > 1:
                    if in_a_subdict:
                        subdict[parts[0].strip(":")] = " ".join(parts[1:])
                        # TODO make this more general; right now this is okay bc we know this can only happen once
                        metadata[subdict_label] = deepcopy(subdict)
                        subdict = {}
                        subdict_label = ""
                        in_a_subdict = False
                    else:
                        metadata[parts[0].strip(":")] = " ".join(parts[1:])
                elif len(parts) == 1:
                    in_a_subdict = True
                    subdict_label = line.strip(":")
            elif in_anno_layers:
                annotation_layers.append(line.strip())
    return metadata, annotation_layers

