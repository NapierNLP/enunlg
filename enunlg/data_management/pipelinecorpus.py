from typing import Callable, Iterable, List, Optional, Union

import enunlg.data_management.iocorpus


class PipelineCorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, metadata: Optional[dict] = None):
        super().__init__(seq)
        if metadata is None:
            self.metadata = {'name': None,
                             'splits': None,
                             'directory': None,
                             'annotation_layers': None
                             }