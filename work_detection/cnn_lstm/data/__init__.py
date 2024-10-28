import os

from torch.utils.data import DataLoader

from data import fudagami


def make_data_loader(args, **kwargs):
    if args.dataset == 'fudagami':
        num_class = 12
        if args.train_mode:
            train_set = fudagami.FudagamiDataset(args, split='train')
            val_set = fudagami.FudagamiDataset(args, split='val')
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None
            population = train_set.population
        else:
            test_set = fudagami.FudagamiDataset(args, split='test')
            train_loader = None
            val_loader = None
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            population = None
        
        return train_loader, val_loader, test_loader, num_class, population

    else:
        raise NotImplementedError

