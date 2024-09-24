import json
from pathlib import Path
from typing import Dict, Union

import logging
import sys
import time

import omegaconf
import hydra
import SPARQLWrapper

from enunlg.data_management.loader import load_data_from_config

import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger('enunlg-scripts.delex_dict_builder')

SUPPORTED_DATASETS = {"enriched-e2e", "enriched-webnlg"}


def delexicalise_with_sem_classes(pipeline_corpus: "enunlg.data_management.enriched_webnlg.EnrichedWebNLGCorpus",
                                  sem_class_dict: Dict[str, str]) -> "enunlg.data_management.enriched_webnlg.EnrichedWebNLGCorpus":
    present = set()
    absent = set()
    for entry in pipeline_corpus:
        # check if entities are in sem_class_data
        logger.debug("-=-=-=-=-=-=-=-==-")
        logger.debug(entry)
        for reference in entry.references.sequence:
            entity = reference.entity
            logger.debug(f"===> entity: {entity}")
            logger.debug(f"---> original tag: {entry.references.entity_orig_tag_mapping[entity]}")
            if entity.lower() in sem_class_dict:
                dbpedia_class = sem_class_dict[entity.lower()]
                entry.delex_reference(entity, dbpedia_class)
                present.add(entity)
            else:
                print(entity)
                # print(entry['lexicalisation'])
                entry['lexicalisation'] = entry['lexicalisation'].replace(reference.orig_delex_tag, str(reference.form), 1)
                # print(entry['lexicalisation'])
                absent.add(entity)
            # if we found one, create a new dict entry mapping the old class to the new one
            # incorporate these changes into extract_reg_from_lex so so we can call the new
            #   method in raw_to_usable to get what we need
    logger.info(f"Percentage of entities for which we have an entry in the sem_class_dict: {len(present) / (len(present) + len(absent))}")
    return pipeline_corpus


def get_dbo_depths_from_txt():
    with Path("datasets/raw/dbpedia_ontology.txt").open('r') as dbpedia_file:
        dpo_dict = {}
        for line in dbpedia_file:
            dpo_dict[line.strip()] = line.count(" ")
        return dpo_dict


class DBPediaSPARQLWrapper(SPARQLWrapper.SPARQLWrapper):
    def __init__(self):
        super().__init__("http://dbpedia.org/sparql")
        self.setReturnFormat(SPARQLWrapper.JSON)
        self.dbo_namespace = "http://dbpedia.org/ontology"
        self.dbo_depths = get_dbo_depths_from_txt()

    def get_dbpedia_class(self, entity, narrowest_class_only=True):
        self.setQuery(f"""PREFIX dbo: <{self.dbo_namespace}>
        SELECT * WHERE {{{{ <http://dbpedia.org/resource/{entity.replace(" ", "_").replace('"', '')}> a ?semclass.}}
        FILTER strstarts(str(?semclass), str(dbo:))}}""")
        try:
            ret = self.queryAndConvert()
            results = [r['semclass']['value'].lstrip(f"{self.dbo_namespace}/") for r in ret["results"]["bindings"]]
            if results:
                if narrowest_class_only:
                    if len(results) == 1:
                        return results[0]
                    else:
                        if all(x in self.dbo_depths for x in results):
                            return max(results, key=lambda x: self.dbo_depths[x])
                        else:
                            # Just return whatever's first in the list if we can't find anything else
                            return results[0]
                else:
                    return results
            else:
                return None
        except SPARQLWrapper.SPARQLExceptions.QueryBadFormed:
            logger.info(f"Badly formed query for entity: {entity}")
        except Exception as e:
            raise


def build_dict(config: omegaconf.DictConfig) -> Union[Dict[str, str], Dict[str, list]]:
    pipeline_corpus = load_data_from_config(config, splits=["train"])

    if config.corpus.name == "e2e-enriched":
        enunlg.data_management.enriched_e2e.validate_enriched_e2e(pipeline_corpus)

    entities = set()
    for entry in pipeline_corpus:
        for reference in entry.references.sequence:
            entities.add(reference.entity)

    logger.info(f"Corpus contains {len(entities)} entities.")
    dbpedia = DBPediaSPARQLWrapper()
    sem_class_dict = {}
    present = 0
    absent = 0
    numbers = 0
    for entity in entities:
        # print('calling fetch function')
        sc = dbpedia.get_dbpedia_class(entity)
        if sc is None:
            absent += 1
            try:
                float_str = str(float(entity))
                int_str = str(int(entity))
                if entity == float_str or entity == int_str:
                    sc = "Number"
                    sem_class_dict[entity] = sc
                    numbers += 1
            except ValueError as e:
                if str(e).startswith("could not convert string to ") or str(e).startswith("invalid literal for int() with base 10"):
                    pass
                else:
                    print(e)
                    raise

        else:
            sem_class_dict[entity] = sc
            present += 1
        print(f"{entity}: {sc}")
        time.sleep(1)
    logger.info(f"Proportion of entities we found a semclass for: {present/(present+absent)}")
    return sem_class_dict


@hydra.main(version_base=None, config_path='../config/data', config_name='enriched-webnlg_as-rdf')
def delex_dict_builder_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    sem_class_dict = build_dict(config)

    json.dump(sem_class_dict, Path('my_delex_dict.tmp').open('w'))


if __name__ == "__main__":
    delex_dict_builder_main()