from pathlib import Path
from typing import List

import logging
import random

import pandas as pd

from enunlg.data_management.pipelinecorpus import TextPipelineCorpus


logger = logging.getLogger('enunlg-scripts.corpus_to_humeval_csv')

corpus_files = {
    'singletask_webnlg_sv': 'outputs/2024-06-02/23-28-16/evaluation-output.corpus',
    'singletask_webnlg_sv_role-delex': 'outputs/2024-06-03/14-41-21/evaluation-output.corpus',
    'singletask_webnlg_rdf': 'outputs/2024-06-02/23-28-21/evaluation-output.corpus',
    'singletask_webnlg_rdf_role-delex': 'outputs/2024-06-02/23-39-09/evaluation-output.corpus',
    'llm_webnlg_sv': "for-analysis/llm/webnlg_slot-value.txt",
    'llm_webnlg_rdf': "for-analysis/llm/webnlg_rdf.txt",
    'singletask_e2e_sv': 'outputs/2024-06-02/22-49-07/evaluation-output.corpus',
    'singletask_e2e_rdf': 'outputs/2024-06-02/23-20-53/evaluation-output.corpus',
    'llm_e2e_sv': "for-analysis/llm/e2e_slot-value.txt",
    'llm_e2e_rdf': "for-analysis/llm/e2e_rdf.txt",
    'ref_webnlg': "for-analysis/enriched-webnlg_refs.txt",
    'ref_e2e': "for-analysis/enriched-e2e_refs.txt",
}


def sample_ids() -> List[str]:
    sample = []
    for idx in range(1, 500, 10):
        i2 = random.choice((1, 2, 3))
        print(f"Id{idx}-Id{i2}")
        sample.append(f"Id{idx}-Id{i2}")
    return sample


if __name__ == "__main__":
    # Load all the results corpora
    corpora_for_analysis = {}
    for sys_corpus_format_delex in corpus_files:
        if corpus_files[sys_corpus_format_delex] is not None:
            corpus_fp = Path(corpus_files[sys_corpus_format_delex])
            corpora_for_analysis[sys_corpus_format_delex] = TextPipelineCorpus.load(corpus_fp)

    sampled_ids = sample_ids()

    metadata_columns = ["id", "system", "corpus", "format", "delex"]
    dfs_for_analysis = {}
    for key in corpora_for_analysis:
        print(key)
        parts = key.split("_")
        system = parts[0]
        corpus = parts[1]
        if parts[2:]:
            mr_type = parts[2]
        else:
            mr_type = "none"
            delex = "none"
        if parts[3:]:
            delex = parts[3]
        else:
            if corpus == "e2e":
                delex = 'name-near-exact-match'
            else:
                delex = 'dbpedia-ontology-classes'
        df_metadata = [system, corpus, mr_type, delex]
        if "llm" in key:
            delex = "none"
            df_metadata[-1] = delex
            annotation_layers = ["raw_input", "GPT4_output", "Llama_output"]
            layer_column_labels = [f"{format}", "GPT4_output", "Llama_output"]
        elif "ref" in key:
            annotation_layers = ["raw_output"]
            layer_column_labels = ["ref"]
        else:
            annotation_layers = ["raw_input", "best_output_relexed"]
            layer_column_labels = [f"{format}", f"{system}_{corpus}_{delex}"]
        rows = []
        for entry in corpora_for_analysis[key]:
            # if entry.metadata.get('id') in sampled_ids:
            row = [entry.metadata.get('id')] + df_metadata
            for layer_name in annotation_layers:
                row.append(entry[layer_name])
            rows.append(row)
        print(df_metadata)
        dfs_for_analysis[key] = pd.DataFrame(rows, columns=metadata_columns + layer_column_labels)
    print(len(dfs_for_analysis))
    common_ids = set()
    e2e_df = None
    for key in dfs_for_analysis:
        if "e2e" in key:
            df = dfs_for_analysis[key]
            print(df.head())
            id_set = set(df['id'])
            if common_ids:
                print("intersecting")
                common_ids = common_ids.intersection(id_set)
            else:
                common_ids = id_set
            print(len(common_ids))
            if e2e_df is None:
                e2e_df = df
            else:
                e2e_df = e2e_df.merge(df, on=["id"])

    print(len(e2e_df))

