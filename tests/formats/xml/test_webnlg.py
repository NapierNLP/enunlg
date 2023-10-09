import os

import xsdata.exceptions
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_webnlg import EnrichedWebNLGBenchmark

webnlg_directories = ["datasets/raw/2023-Challenge/data", "datasets/raw/webnlg/data/v2.0/en/train"]

for directory in webnlg_directories:
    print(directory)
    if os.path.exists(directory):
        print(directory)
        filenames = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".xml")]
        for fn in filenames:
            try:
                xsparsers.XmlParser().parse(fn, EnrichedWebNLGBenchmark)
            except xsdata.exceptions.ParserError as e:
                print(e.args)
                print(fn)
                raise


