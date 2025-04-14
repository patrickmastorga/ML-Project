import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

import numpy as np
from sklearn.linear_model import SGDClassifier
from pathlib import Path

from celeba_vae_bernoulli import CelebATransform, ConvDecoder, ConvEncoder, VAE
from utils import generate_latent_space_traversals_along_direction

attrs = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
]

if __name__ == "__main__":
    LATENT_DIMS = 64
    MODEL_NAME = 'celeba_vae_64'
    FEATURES = ('Eyeglasses')

    dir_path = Path('models') / MODEL_NAME

    sorted_attrs = [attr for attr in attrs if attr in FEATURES]
    indices = [attrs.index(attr) for attr in sorted_attrs]

    # download MNIST dataset
    print('Loading dataset...')
    data = CelebA(root='./data', split='test', target_type='attr', transform=CelebATransform(), download=True)
    dataloader = DataLoader(dataset=data, batch_size=256, shuffle=False, num_workers=8)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    encoder = ConvEncoder(latent_dims=LATENT_DIMS, base_channels=64).to(device=device)
    decoder = ConvDecoder(latent_dims=LATENT_DIMS, base_channels=64).to(device=device)
    vae = VAE(encoder, decoder).to(device=device)
    vae.load_state_dict(torch.load(dir_path / 'model.pth'))
    vae.eval()

    # get latent representations
    print('Getting latent representations...')
    latents, all_labels = [], []
    with torch.no_grad():
        for images, attr in dataloader:
            images = images.to(device)
            mu, _ = vae.encoder(images)
            latents.append(mu.cpu())
            all_labels.append(attr[:, indices].cpu())
    latents = torch.cat(latents).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # normalize latents
    latents = (latents - latents.mean(axis=0)) / (latents.std(axis=0) + 1e-8)

    for i, attr in enumerate(sorted_attrs):
        labels = all_labels[:, i]

        # fit logistic regression
        print(f'Fitting logistic regression for {attr}...')
        clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4,
                        class_weight='balanced', learning_rate='optimal',
                        max_iter=50, early_stopping=True, n_jobs=-1, verbose=0)
        clf.fit(latents, labels)

        # save direction
        w = clf.coef_.astype(np.float32).ravel()
        direction = w / np.linalg.norm(w)
        #np.save(dir_path / f'{FEATURE}_direction.npy', direction)
        
        # generate latent space traversal along direction
        print(f'Generating latent space traversal for {attr}...')
        generate_latent_space_traversals_along_direction(
            decoder=vae.decoder,
            dir=torch.tensor(direction, dtype=torch.float32),
            path=dir_path / f'{attr}_traversal2.png',
            num_traversals=5,
            size=11,
            imgsize=(3, 4),
            device=device,
        )