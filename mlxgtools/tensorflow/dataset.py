import tensorflow as tf 
from tqdm import tqdm 
from typing import Tuple, List


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if type(value) == str:
        value = value.encode("utf8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class CreateTFRecord(object):
    TYPE_MAP = {
        "int": _int64_feature,
        "float": _float_feature,
        "other": _bytes_feature
    }
    
    
    def __init__(self, feat_type_map: [Tuple]):
        self.feat_type_map = feat_type_map

    
    def __create_feature(self, *args):
        feature = {}
        assert len(self.feat_type_map) == len(args)
        for (name, _type), v in zip(self.TYPE_MAP, args):
            feature[name] = _type(v)
        return feature
    
    def serialize_example(self, *args):
        feature = self.__create_feature(*args)
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def map_fnc(self, row):
        """transform func row > tuple
        Args:
            row ([type]): [source data]
        Returns:
            [tuple]: [seq > map seq]
        """
        return row 
    
    def create_tfrecord_of_pands(self, df, tfrec_save_path, **kwargs):
        with tf.io.TFRecordWriter(tfrec_save_path) as writer:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=kwargs.get('desc', ''), mininterval=kwargs.get('mininterval', 10), leave=kwargs.get('leave', True)):
                row = self.map_fnc(row)
                example = self.serialize_example(row)
                writer.write(example)
                

    def create_tfrecord_of_iterable(self, iterable, tfrec_save_path, **kwargs):
        with tf.io.TFRecordWriter(tfrec_save_path) as writer:
            for row in tqdm(iterable=iterable, total=kwargs.get('total', None), desc=kwargs.get('desc', ''), mininterval=kwargs.get('mininterval', 10), leave=kwargs.get('leave', True)):
                row = self.map_fnc(row)
                example = self.serialize_example(row)
                writer.write(example)
                         
                
                
        


