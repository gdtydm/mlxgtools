## 
import os 
import json 
import copy 
from multiprocessing import Pool
from datetime import datetime 


class ArgsClass(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def delete(self, *args):
        for arg in args:
            delattr(self, arg)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(**json_object)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())



def get_gcs_path_of_kaggle_data(data_name, is_private=False):
    if is_private:
        # Step 1: Get the credential from the Cloud SDK
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        user_credential = user_secrets.get_gcloud_credential()
        # Step 2: Set the credentials
        user_secrets.set_tensorflow_credential(user_credential)

    # Step 3: Use a familiar call to get the GCS path of the dataset
    from kaggle_datasets import KaggleDatasets
    GCS_DS_PATH = KaggleDatasets().get_gcs_path(data_name)	
    return GCS_DS_PATH 


def multiprocess_fnc_without_res(fn, args, core):
    parallel_cnt = 0
    parallel_num = len(args)
    def call_back(rst):
        parallel_cnt += 1
        print('%s, %d / %d done!' % (datetime.datetime.now(), parallel_cnt, parallel_num))

    pool = Pool(core)
    for arg in args:
        pool.apply_async(fn, args=arg,  callback=call_back)
    pool.close()
    pool.join()

def multprocess_fnc(fn, args, core):
    parallel_cnt = 0
    parallel_num = len(args)
    def call_back(rst):
        parallel_cnt += 1
        print('%s, %d / %d done!' % (datetime.datetime.now(), parallel_cnt, parallel_num))


    serverid_num = len(args)
    record_data = []
    record_num = 0 # ###成功运行
    empty_num = 0   # ###成功运行但是结果为空
    error_num = 0  # ###运行失败
    pool = Pool(4)
    for arg in args:
        record_data.append(pool.apply_async(fn, args=arg, callback=call_back))
    pool.close()
    pool.join()

    for i in record_data:
        # ####运行无误
        if i.get()[1] == 0:
            record_num += 1
        # ####运行无误，但是无结果
        if i.get()[1] == 0 and i.get() == 0:
            empty_num += 1

        # ####运行错误
        if i.get()[1] == 1 :
            error_num += 1

    print(f'[INFO] all num: {serverid_num}, record num: {record_num}, empty num: {empty_num}, error num: {error_num}')

    res = [i.get()[0] for i in record_data] 
    return res 

