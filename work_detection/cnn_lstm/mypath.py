class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'factory':
            return '/home/kataoka/dataset/factory_seg/'
        elif dataset == 'person_part':
            return '/groups/gce50899/persons_dataset/'
        elif dataset == 'factory2':
            return '/groups/gce50899/kataoka/dataset/aopa/Factory_Operation_Video_Dataset_V2/'
        elif dataset == 'fudagami':
            return '/groups/gce50899/kataoka/dataset/aopa/deeplab_outputs/factory/deeplab-mobilenet/experiment_1/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
