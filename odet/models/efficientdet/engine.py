import torch
import torch.nn as nn


def train_single_epoch(model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_fn,
                       data_loader: torch.utils.data.DataLoader,
                       epoch: int,
                       device: torch.device,
                       grad_accumulation_steps: int = 1,
                       scheduler = None):
    model.train()
    
    running_loss = 0.0
    running_reg_loss = 0.0
    running_clf_loss = 0.0

    n_steps = 0

    for i, (images, annotations) in enumerate(data_loader):

        images = images.to(device)
        annotations = annotations.to(device)

        classification, regression, anchors = model(images)
        
        clf_loss, reg_loss = loss_fn(classification, regression, 
                                     anchors, annotations)

        clf_loss = clf_loss.mean()
        reg_loss = reg_loss.mean()
        loss = clf_loss + reg_loss

        if loss == 0:
            print('loss equal zero(0)')
            continue

        loss.backward()
        
        if (i + 1) % grad_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        running_clf_loss += clf_loss.item()
        running_reg_loss += reg_loss.item()
        n_steps += 1

        if (i + 1) % 10 == 0:
            loss_mean = running_loss / n_steps
            clf_loss_mean = running_clf_loss / n_steps
            reg_loss_mean = running_reg_loss / n_steps

            print(f'Epoch[{epoch}] [{i}/{len(data_loader)}]'
                  f'  loss: {loss_mean:.3f}'
                  f'  clf_loss: {clf_loss_mean:.3f}'
                  f'  reg_loss: {reg_loss_mean:.3f}')

    if scheduler is not None:
        scheduler.step(running_loss / n_steps)
