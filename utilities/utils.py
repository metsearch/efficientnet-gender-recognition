import os
import random

import numpy as np
import seaborn as sns
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from .log import logger

def tensor2img(t, t_type = 'rgb'):    
    gray_transformations = transforms.Compose([
        transforms.Normalize(mean=[0.], std = [1/0.5]),
        transforms.Normalize(mean=[-0.5], std=[1])
    ])
    
    rgb_transformations = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    
    invTrans = gray_transformations if t_type == "gray" else rgb_transformations 
    return (invTrans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def display_images_with_predictions(path2metrics, images, predictions, num_images=25, num_images_per_row=5):    
    plt.figure(figsize = (20, 10))
    images_to_display_idxes = [random.randint(0, len(images) - 1) for _ in range(num_images)]
    for i, idx in enumerate(images_to_display_idxes):
        img = images[idx].squeeze()
        img = tensor2img(images[idx])
        plt.subplot(num_images_per_row, num_images // num_images_per_row, i + 1)
        plt.imshow(img); plt.axis('off')
        plt.title(f'{predictions[idx]}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(path2metrics, f'inference_examples.png'))
    plt.show()

def plot_confusion_matrix(path2metrics, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(path2metrics, f'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    logger.info('Testing utils...')