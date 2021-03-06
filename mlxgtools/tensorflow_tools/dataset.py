import tensorflow as tf 
from tqdm import tqdm 
from typing import Tuple, List, Dict
<<<<<<< HEAD
from functools import partial
=======
from abc import ABCMeta, abstractmethod

>>>>>>> 0055020036b954747b3e03309c8dfbf36337827e

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
                         

<<<<<<< HEAD

def read_tfrecord_fixed_feat(example, type_dict:Dict, map_fnc) -> Dict:
    def create_tfrec_format(type_dict):
        LABELED_TFREC_FORMAT = {}
        for k,v in type_dict:
            LABELED_TFREC_FORMAT[k] = tf.io.FixedLenFeature([], v)
        return LABELED_TFREC_FORMAT

    LABELED_TFREC_FORMAT = create_tfrec_format(type_dict)
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    res = map_fnc(example)
    return res


def load_tfrec_dataset(filenames, parse_tfrec_fn, ordered=False, AUTO=tf.data.experimental.AUTOTUNE):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order,increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(parse_tfrec_fn, num_parallel_calls=AUTO)
    return dataset

def get_tfrecord_training_dataset(filenames, parse_tfrec_fn, batch_size, map_fn, AUTO=tf.data.experimental.AUTOTUNE):
    dataset = load_tfrec_dataset(filenames, parse_tfrec_fn,ordered=False, AUTO=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    if map_fn is not None:
        dataset = dataset.map(map_fn, num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_tfrecord_validation_dataset(filenames, parse_tfrec_fn, batch_size, cache=False, AUTO=tf.data.experimental.AUTOTUNE):
    dataset = load_tfrec_dataset(filenames, parse_tfrec_fn,ordered=True, AUTO=AUTO)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    if cache:
        dataset = dataset.cache()
    return dataset

=======
class ReadTFRecFile(object):
    def __init__(self, type_dict, **kwargs):
        self.type_dict = type_dict
        self.batch_size = kwargs.get(batch_size, 64)
        self.auto = tf.data.experimental.AUTOTUNE 

    def read_tfrecord_fixed_feat(self, example):
        def create_tfrec_format():
            LABELED_TFREC_FORMAT = {}
            for k,v in self.type_dict:
                LABELED_TFREC_FORMAT[k] = tf.io.FixedLenFeature([], v)
            return LABELED_TFREC_FORMAT

        LABELED_TFREC_FORMAT = create_tfrec_format()
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        return example

    def map_fn(self, example):
        return example 
    
    def load_tfrec_dataset(self, filenames, ordered=False):
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order,increase speed
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.auto) # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(self.read_tfrecord_fixed_feat, num_parallel_calls=self.auto)
        return dataset

    def get_dataset(self, filenames, **kwargs):
        dataset = self.load_tfrec_dataset(filenames,ordered=kwargs.get('ordered', True))

        if kwargs.get('repeat', False):
            dataset = dataset.repeat() # the training dataset must repeat for several epochs 

        batch_size =  kwargs.get('batch_size',None)
        if batch_size  is None:
            batch_size = self.batch_size

        dataset = dataset.map(self.map_fn)
        if kwargs.get('shuffle', False):
            dataset = dataset.shuffle(kwargs.get('shuffle_size', batch_size*10))

        dataset = dataset.batch(batch_size) 
        dataset = dataset.prefetch(self.auto)
        if kwargs.get('cache', False):
            dataset = dataset.cache()
        return dataset

    def get_train_dataset(self, filenames, ordered=False, shuffle=True, repeat=True, cache=False, batch_size=None, **kwargs):
        dataset = self.get_dataset(filenames, ordered=ordered, shuffle=shuffle, repeat=repeat, cache=cache, batch_size=batch_size, **kwargs)
        return dataset

    def get_valid_dataset(self, filenames, ordered=True, shuffle=False, repeat=False, cache=True, batch_size=None, **kwargs):
        dataset = self.get_dataset(filenames, ordered=ordered, shuffle=shuffle, repeat=repeat, cache=cache, batch_size=batch_size, **kwargs)
        return dataset

    def check_parsed(self, dataset, n=5):
        for example in dataset.take(n):
            print(">>>" * 7)
            print(example)
            print("<<<" * 7)
>>>>>>> 0055020036b954747b3e03309c8dfbf36337827e
