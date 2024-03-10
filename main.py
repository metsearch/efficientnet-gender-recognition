import os
import pickle

import click
from rich.progress import track
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from dataset import CelebADataset

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='Enable debug mode', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug
    invoked_subcommand = ctx.invoked_subcommand
    if invoked_subcommand is None:
        logger.info('No subcommand was specified')
    else:
        logger.info(f'Invoked subcommand: {invoked_subcommand}')

@router_cmd.command()
@click.option('--path2source', help='Path to source data', required=True)
@click.option('--path2destination', help='Path to data', default='data/')
def grabber(path2source, path2destination):
    logger.debug('Grabbing data...')
    if not os.path.exists(path2destination):
        os.makedirs(path2destination)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = {}    
    for data_type in ['train', 'val', 'test']:
        current_dataset = CelebADataset(data_source=path2source, data_type=data_type, transform=transform)
        dataset[data_type] = len(current_dataset)
        with open(os.path.join(path2destination, f'{data_type}_dataset.pkl'), 'wb') as f:
            pickle.dump(current_dataset, f)
    
    print(f'Dataset: {dataset}')
    logger.info('All data grabbed!')
    
@router_cmd.command()
@click.option('--path2data', help='Path to data', default='data/')
@click.option('--path2models', help='Path to models', default='models/')
@click.option('--path2metrics', help='Path to metrics', default='metrics/')
@click.option('--bt_size', help='Batch size', default=64)
@click.option('--num_epochs', help='Number of epochs', default=10)
def learn(path2data, path2models, bt_size, num_epochs, path2metrics):
    logger.debug('Training...')
    if not os.path.exists(path2models):
        os.makedirs(path2models)    
    if not os.path.exists(path2metrics):
        os.makedirs(path2metrics)
        
    with open(os.path.join(path2data, 'train_dataset.pkl'), 'rb') as f:
        train_dataset = pickle.load(f)
    train_loader = DataLoader(train_dataset, batch_size=bt_size, shuffle=True)
    
    with open(os.path.join(path2data, 'val_dataset.pkl'), 'rb') as f:
        val_dataset = pickle.load(f)
    val_loader = DataLoader(val_dataset, batch_size=bt_size, shuffle=False)
    
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    
    pretrained_efficientnet = models.efficientnet_b0(weights='DEFAULT')
    pretrained_efficientnet = pretrained_efficientnet.to(device)
    
    for param in pretrained_efficientnet.parameters():
        param.requires_grad = False
    
    pretrained_efficientnet.classifier[1] = nn.Linear(pretrained_efficientnet.classifier[1].in_features, out_features=2)
    pretrained_efficientnet = pretrained_efficientnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_efficientnet.classifier[1].parameters(), lr=0.001)
    
    losses = []
    nb_data = len(train_loader)
    for epoch in range(num_epochs):
        counter = 0
        epoch_loss = 0.0
        pretrained_efficientnet.train()
        for X, Y in track(train_loader, description='Training'):
            inputs = X.to(device)
            labels = Y.to(device)
            optimizer.zero_grad()
            outputs = pretrained_efficientnet(inputs)
            E: th.Tensor = criterion(outputs, labels)
            E.backward()
            optimizer.step()
            
            epoch_loss += E.cpu().item()
            counter += len(inputs)
        
        average_loss = epoch_loss / nb_data
        losses.append(average_loss)
        logger.debug(f'[{epoch:03d}/{num_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {average_loss:07.3f}')

    th.save(pretrained_efficientnet.cpu(), os.path.join(path2models, 'model.pth'))
    logger.info('The model was saved ...!')
    
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path2metrics, 'training_loss.png'))
    plt.show()
    
@router_cmd.command()
@click.option('--path2models', help='path to models', type=click.Path(True), default='models/')
def predict(path2models, bt_size):
    logger.debug('Inference...')

    path2metrics = 'metrics/'
    if not os.path.exists(path2metrics):
        os.makedirs(path2metrics)
    
    model = th.load(os.path.join(path2models, 'model.pth'))
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    
    with open(os.path.join('data', 'test_dataset.pkl'), 'rb') as f:
        test_dataset = pickle.load(f)
    test_loader = DataLoader(test_dataset, batch_size=bt_size, shuffle=True)
    
    model.eval()
    
    predictions = []
    ground_truths = []
    images = []
    
    with th.no_grad():
        for X, Y in track(test_loader, description='Inference...'):            
            X = X.to(device)
            Y = Y.to(device)

            P = model(X)
            predictions.extend(th.argmax(P, dim=1).cpu().numpy())
            ground_truths.extend(Y.cpu().numpy())
            images.extend(X.cpu().numpy())

    model.train()
    
    display_images_with_predictions(path2metrics, images, predictions)
    plot_confusion_matrix(path2metrics, ground_truths, predictions, list(range(10)))
    
if __name__ == '__main__':
    logger.info('...')
    router_cmd(obj={})