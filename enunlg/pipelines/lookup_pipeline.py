import enunlg.data_management.enriched_e2e as ee2e
import enunlg.templates.lookup as lug


if __name__ == "__main__":
    corpus = ee2e.load_enriched_e2e(splits=("train", ))
    for x in corpus.entries[:5]:
        print(x)
