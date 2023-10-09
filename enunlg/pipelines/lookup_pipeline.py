from dataclasses import asdict

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.templates.lookup as lug


if __name__ == "__main__":
    corpus = ee2e.load_enriched_e2e(splits=("dev", ))
    for x in corpus[:6]:
        print(x)
