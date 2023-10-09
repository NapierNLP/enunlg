from dataclasses import asdict

import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.templates.lookup as lug


class PipelineLookupGenerator(object):
    def __init__(self, corpus: ee2e.EnrichedE2ECorpus):
        self.layers = corpus.layers
        self.pipeline = corpus.views
        self.modules = {layer_pair: lug.OneToManyLookupGenerator() for layer_pair in self.pipeline}
        for layer_pair in self.modules:
            self.modules[layer_pair].train(corpus.items_by_view(layer_pair))

    def predict(self, mr):
        curr_input = mr
        for layer_pair in self.modules:
            logging.debug(layer_pair)
            logging.debug(curr_input)
            curr_output = self.modules[layer_pair].predict(curr_input)
            logging.debug(curr_output)
            curr_input = curr_output
        return curr_output


if __name__ == "__main__":
    corpus = ee2e.load_enriched_e2e(splits=("dev", ))
    for x in corpus[:6]:
        print(x)

    plg = PipelineLookupGenerator(corpus)
    for entry in corpus:
        mr = entry.raw_input
        print(mr)
        print(plg.predict(mr))
        print("----")
