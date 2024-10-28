import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'factory_operation_video_dataset_v2':
            return os.getenv('DATASET_DIR', '/path/to/default/dataset')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
