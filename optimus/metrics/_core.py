"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""


class Metric:
    def __init__(self, name: str, maximise: bool = True):
        self.name = name
        self.maximise = maximise

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
