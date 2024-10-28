class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'fudagami':
            return '/groups/gce50899/kataoka/dataset/aopa/deeplab_outputs/factory/deeplab-mobilenet/experiment_1/'
        if dataset == 'factory_operation_video_dataset_v2':
            return '/groups/gce50899/kataoka/dataset/aopa/Factory_Operation_Video_Dataset_V2/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError



