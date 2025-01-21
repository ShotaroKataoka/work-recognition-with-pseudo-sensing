from torchvision import transforms

def get_transforms(config, is_train=True):
    transform_list = [transforms.Resize(tuple(config['train_transforms']['resize']))]

    if is_train and config['train_transforms']['random_horizontal_flip']:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=config['train_transforms']['normalize']['mean'],
            std=config['train_transforms']['normalize']['std']
        )
    )

    return transforms.Compose(transform_list)
