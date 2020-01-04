import time
import click
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import odet.data as data_utils
import odet.models.efficientdet as efficientdet
from .augmentation import get_augmentation


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = ['treecko', 'mewtwo', 'greninja', 'psyduck', 'solgaleo']


def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = torch.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = torch.FloatTensor(annot)
                annot_padded[idx, :len(annot), 4] = torch.FloatTensor(lab)
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))


def create_datasets(ds_root: str, 
                    images_root: str,
                    net: str) -> torch.utils.data.Dataset:
    net = efficientdet.EFFICIENTDET[net]
    tfms = get_augmentation(phase='train', 
                             width=net['input_size'], 
                             height=net['input_size'])
    train_dataset = data_utils.LabelMeDataset(
        root=ds_root,
        images_path=images_root, 
        classes=CLASSES,
        transforms=tfms)
    return train_dataset


def load_model(n_classes: int, 
               pretrained_path: str,
               checkpoint_path: str, 
               net: str) -> torch.nn.Module:

    checkpoint = None
    pretrained = None

    if checkpoint_path is not None:
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        loaded_n_classes = checkpoint['num_class']
        net = checkpoint['network']

    elif pretrained_path is not None:
        print("Loading pretrained: {} ...".format(pretrained_path))
        pretrained = torch.load(pretrained_path, map_location=DEVICE)
        net = pretrained['network']
        loaded_n_classes = pretrained['num_class']
    else:
      loaded_n_classes = n_classes

    net_opts = efficientdet.EFFICIENTDET[net]
    model = efficientdet.EfficientDet(num_classes=loaded_n_classes,
                                      network=net,
                                      W_bifpn=net_opts['W_bifpn'],
                                      D_bifpn=net_opts['D_bifpn'],
                                      D_class=net_opts['D_class'])
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
    
    if pretrained is not None:
        model.load_state_dict(pretrained['state_dict'])
        model.bbox_head.num_classes = n_classes
        model.bbox_head.retina_cls = torch.nn.Conv2d(
            model.bbox_head.feat_channels,
            model.bbox_head.num_anchors * n_classes,
            3, padding=1)
        
    model.to(DEVICE)

    return model


def train(**args):

    ds = create_datasets(args['path'], args['images_path'], args['network'])

    train_dataloader = DataLoader(ds,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    
    model = load_model(args['num_classes'], 
                       args['pretrained'],
                       args['checkpoint'],
                       args['network'])
    

    optimizer = optim.AdamW(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, verbose=True)
    criterion = efficientdet.FocalLoss()

    checkpoint_dir = Path(args['save'])
    model.train()
    iteration = 1
    for epoch in range(args['epochs']):
        print("{} epoch: \t start training....".format(epoch))
        start = time.time()
        
        result = {}
        total_loss = 0.
        steps = 0
        optimizer.zero_grad()
        for idx, (images, annotations) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            annotations = annotations.to(DEVICE)
            classification, regression, anchors = model(images)
            classification_loss, regression_loss = criterion(
                classification, regression, anchors, annotations)
            
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            
            if bool(loss == 0):
                print('loss equal zero(0)')
                continue
            loss.backward()

            # Check if we have to backprop the gradients
            if (idx+1) % args['grad_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1
            if iteration % 100 == 0:
                print('{} iteration: training ...'.format(iteration))
                ans = {
                    'epoch': epoch,
                    'iteration': iteration,
                    'cls_loss': classification_loss.item(),
                    'reg_loss': regression_loss.item(),
                    'mean_loss': total_loss / steps
                }
                for key, value in ans.items():
                    print('    {:15s}: {}'.format(str(key), value))

            iteration += 1
        scheduler.step(total_loss / steps)
        result = {
            'time': time.time() - start,
            'loss': total_loss / steps
        }
        for key, value in result.items():
            print('    {:15s}: {}'.format(str(key), value))
        arch = type(model).__name__
        state = {
            'arch': arch,
            'num_class': args['num_classes'],
            'network': args['network'],
            'state_dict': model.state_dict()
        }
        fname = 'efficientdet_{}_{}.pt'.format(args['network'], epoch)
        chkp_path = str(checkpoint_dir / fname)
        torch.save(state, chkp_path)
    state = {
        'arch': arch,
        'num_class': args['num_classes'],
        'network': args['network'],
        'state_dict': model.state_dict()
    }
    fname = 'Final_efficientdet_{}_{}.pt'.format(args['network'], epoch)
    chkp_path = str(checkpoint_dir / fname)
    torch.save(state, chkp_path)


@click.command()
@click.option('--path', type=click.Path(file_okay=False, exists=True))
@click.option('--images-path', type=click.Path(file_okay=False, exists=True))
@click.option('--network', default='efficientdet-d0', type=str,
              help='efficientdet-[d0, d1, ..]')
@click.option('--checkpoint', default=None, 
              type=click.Path(dir_okay=False),
              help='Checkpoint state_dict file to resume training from')
@click.option('--pretrained', default=None,
              type=click.Path(file_okay=True, exists=False))
@click.option('--epochs', default=20, type=int,
              help='Num epoch for training')
@click.option('--batch-size', default=16, type=int,
               help='Batch size for training')
@click.option('--num-classes', default=5, type=int,
               help='Number of class used in model')
@click.option('--grad_accumulation_steps', default=1, type=int,
               help='Number of gradient accumulation steps')
@click.option('--lr', '--learning-rate', default=1e-4, type=float,
               help='initial learning rate')
@click.option('--save', default='models/efficientdet', 
              type=click.Path(file_okay=False, exists=True),
              help='Directory for saving checkpoint models')
def main(**args):
    train(**args)


if __name__ == '__main__':
    main()
