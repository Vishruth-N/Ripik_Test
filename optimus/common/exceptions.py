"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""


class SequenceError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
