import os
from dotenv import load_dotenv, find_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv(find_dotenv())

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'factory_operation_video_dataset_v2':
            dataset_dir = os.getenv('DATASET_DIR')
            if dataset_dir:
                return dataset_dir
            else:
                raise EnvironmentError('環境変数 DATASET_DIR が設定されていません。')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
