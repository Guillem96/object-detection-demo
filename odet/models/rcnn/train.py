import click
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import odet.data as data_utils
import odet.models.rcnn as rcnn
import odet.models.rcnn.engine as engine


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _sample(labels: torch.LongTensor,
            sample_size: int = 64) -> torch.LongTensor:

    if labels.size(0) < sample_size:
        # Just return random shuffled indices
        return torch.randperm(labels.size(0))
    
    foreground_indices = torch.nonzero(labels).squeeze(-1)
    background_indices = torch.nonzero((labels == 0).long()).squeeze(-1)

    true_instances = int(sample_size * 0.25)
    false_instances = sample_size - true_instances
    
    n_true_instances = foreground_indices.size(0)
    n_false_instances = background_indices.size(0)

    if n_true_instances > 0:
        rand_foreground_idx = torch.randint(high=n_true_instances, 
                                            size=(true_instances,))
    else:
        rand_foreground_idx = torch.LongTensor([])
        false_instances = sample_size

    rand_background_idx = torch.randint(high=n_false_instances,
                                        size=(false_instances,))

    # Concatenate all random samples
    idx = torch.cat([foreground_indices[rand_foreground_idx],
                     background_indices[rand_background_idx]], dim=0)
    
    # Shuffle and return the random sample
    return idx[torch.randperm(idx.size(0))]


def collate_fn(batch):
    images, annots = zip(*batch)

    # Biased sample
    labels = []
    rt = []
    boxes = []
    for i in range(len(annots)):
        current_labels =  annots[i]['labels']
        sample_idx = _sample(current_labels)
        labels.append(current_labels[sample_idx])
        boxes.append(annots[i]['ss_boxes'][sample_idx])
        rt.append(annots[i]['regression_targets'][sample_idx])

    return (torch.stack(images), 
            (torch.stack(boxes), torch.stack(rt), torch.stack(labels)))


def loss_fn(y_trues: torch.LongTensor, 
            y_pred: torch.FloatTensor,
            reg_targets: torch.FloatTensor,
            reg_preds: torch.FloatTensor):

    clf_loss = F.cross_entropy(y_pred, y_trues)

    reg_targets = reg_targets.view(-1, 4)
    reg_preds = reg_preds.view(-1, 4)
    y_trues = y_trues.view(-1)

    reg_mask = y_trues != 0 # Only regress true boxes
    reg_loss = F.smooth_l1_loss(reg_preds[reg_mask], 
                                reg_targets[reg_mask])
    
    return clf_loss, reg_loss


def train(**args):
    save_path = Path(args['save'])

    classes = ['treecko', 'mewtwo', 'greninja', 'psyduck', 'solgaleo']

    ds = data_utils.RCNNDataset(args['annots_path'], 
                                classes=classes, 
                                transforms=engine.get_transforms('train', (224, 224)))
    train_size = int(len(ds) * 0.9)
    rand_idx = torch.randperm(len(ds)).long()

    train_ds = Subset(ds, rand_idx[:train_size])
    test_ds = Subset(ds, rand_idx[train_size:])

    train_dl = DataLoader(train_ds, 
                          batch_size=args['batch_size'],
                          shuffle=True,
                          collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, 
                         batch_size=args['batch_size'],
                         shuffle=False,
                         collate_fn=collate_fn)

    model = rcnn.RCNN(feature_extractor_freeze=bool(args['freeze']), 
                      n_classes=len(classes) + 1) 
    model.to(DEVICE)

    if args['checkpoint']:
        print('Loading model from existing checkpoint')
        chkp = torch.load(args['checkpoint'], map_location=DEVICE)
        model.load_state_dict(chkp)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, 
                           lr=args['learning_rate'])

    for epoch in range(args['epochs']):
        engine.train_single_epoch(model=model, 
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  data_loader=train_dl, 
                                  epoch=epoch,
                                  device=DEVICE)
        
        engine.evaluate(model=model, 
                        loss_fn=loss_fn, 
                        data_loader=test_dl,
                        device=DEVICE)

        model_fname = f'rcc_{epoch}.pt'
        torch.save(model.state_dict(), str(save_path / model_fname))


@click.command()
@click.option('--epochs', type=int, default=5)
@click.option('--batch-size', type=int, default=2)
@click.option('--learning-rate', type=float, default=1e-3)

@click.option('--freeze/--no-freeze', default=False)

@click.option('--annots-path', 
              type=click.Path(exists=True, file_okay=False))

@click.option('--checkpoint',
              default='',
              type=click.Path(dir_okay=False))
@click.option('--save',
              default='models/',
              type=click.Path(dir_okay=True, exists=True))
def main(**args):
    train(**args)


if __name__ == "__main__":
    main()