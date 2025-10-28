import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar library
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import yaml, os
import utils.models as models
import utils.training_scripts as training_scripts
from utils import evaluation_utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import ImageOps, ImageFilter
from torchvision.transforms import InterpolationMode
import shutil
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

torch.autograd.set_detect_anomaly(True)


if dist.is_initialized():
    dist.destroy_process_group()

def init_process_group():
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())


init_process_group()
world_size = torch.cuda.device_count()



torch.manual_seed(35)
np.random.seed(35)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
torch.backends.cudnn.benchmark = True


# Access hyperparameters
###############################################################



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


batch_size = config['hyperparameters']['batch_size']
learning_rate = config['hyperparameters']['learning_rate']
num_epochs = config['hyperparameters']['num_epochs']
classifier_epochs = config['hyperparameters']['classifier_epochs']
latent_dim = config['hyperparameters']['latent_dim']
expander_dim = config['hyperparameters']['expander_dim']
data_set_name = config['dataset']['data_set_name']
test_using_classifier = config['hyperparameters']['test_using_classifier']
number_of_clusters = config['dataset']['number_of_clusters']
number_of_workers = config['dataset']['number_of_workers']
prefetch_factor = config['dataset']['prefetch_factor']
weight_decay = config['hyperparameters']['weight_decay']
optimizer = config['hyperparameters']['optimizer']
EXPANDER_HIDDEN_DIM = config['hyperparameters']['EXPANDER_HIDDEN_DIM']
DATASET_NUMBER_OF_CLASS = config['dataset']['DATASET_NUMBER_OF_CLASS']
MODEL_PATH = config['hyperparameters']['MODEL_PATH']
LOAD_THE_MODEL = config['hyperparameters']['LOAD_THE_MODEL']
SAVE_THE_MODEL = config['hyperparameters']['SAVE_THE_MODEL']
LOG_PATH = config['hyperparameters']['LOG_name']



CONFIG_PATH = "config.yaml"  
config_save_path = os.path.join(f"logs/{LOG_PATH}/config/", "config.yaml")
os.makedirs(config_save_path, exist_ok=True)
shutil.copy(CONFIG_PATH, config_save_path)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

transform_MNIST = transforms.Compose([
    transforms.RandomRotation(20),  # Rotate the image by up to 20 degrees
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random translations
    # transforms.RandomHorizontalFlip(),  # Add horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random color jitter
    transforms.ToTensor(),  # Convert to Tensor
])

transform_CIFAR = transforms.Compose([
    # # transforms.RandomResizedCrop(32, scale=(0.2, 1.0)), 
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    # transforms.RandomRotation(15),
    # transforms.RandomGrayscale(p=0.2),
    # # transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 0.5)),
    # transforms.ToTensor(),
    # # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.ToTensor(),
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(0.2),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
    transforms.RandomSolarize(0.5, p=0.2),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform_STL = transforms.Compose([
    # transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), 
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    # transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2240, 0.2215, 0.2239)),
])

transform_TINY_IMAGENET = transforms.Compose([
    # transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), 
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# transform_MINI_IMAGENET = transforms.Compose([
#     # transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), 
#     # transforms.RandomHorizontalFlip(),
#     transforms.Resize((84, 84)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# ])

transform_MINI_IMAGENET = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.RandomResizedCrop(84, scale=(0.6, 0.8), interpolation=InterpolationMode.BICUBIC),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    GaussianBlur(p=.1),
    transforms.RandomRotation(5),
    # Solarization(p=0.0),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


transform_map = {
    'MNIST': transform_MNIST,
    'CIFAR10': transform_CIFAR,
    'STL10': transform_STL,
    'TINY_IMAGENET': transform_TINY_IMAGENET,
    'MINI_IMAGENET': transform_MINI_IMAGENET
}
model_map = {
    'MNIST': models.Encoder(latent_dim=latent_dim).to(device),
    # 'CIFAR10': Models.ResNetEncoder(latent_dim=latent_dim).to(device)
    'CIFAR10': models.ResNetEncoder(latent_dim=latent_dim).to(device),
    # 'STL10': Models.MobileNetEncoder(latent_dim=latent_dim).to(device)
    # 'CIFAR10': models.Encoder_cifar(latent_dim=latent_dim).to(device),
    'STL10': models.Encoder_stsl(latent_dim=latent_dim).to(device),
    'TINY_IMAGENET': models.Encoder_tiny_imagenet(latent_dim=latent_dim).to(device),
    # 'MINI_IMAGENET': models.ResNet34FeatureExtractor(latent_dim=latent_dim).to(device)
}
###############################################################

# Load the dataset 
###############################################################
if data_set_name == 'MNIST':    
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_map[data_set_name], download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_map[data_set_name], download=True)

    sampler = DistributedSampler(
        train_dataset,  # Pass the dataset here
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=number_of_workers,  prefetch_factor=prefetch_factor, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=number_of_workers,  prefetch_factor=prefetch_factor, sampler=sampler)

if data_set_name == 'CIFAR10':    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_map[data_set_name], download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_map[data_set_name], download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=number_of_workers,  prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=number_of_workers,  prefetch_factor=prefetch_factor)

if data_set_name == 'STL10':
    train_dataset = datasets.STL10(root='./data', split='train', transform=transform_map[data_set_name], download=False)
    test_dataset = datasets.STL10(root='./data', split='test', transform=transform_map[data_set_name], download=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=number_of_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=number_of_workers, prefetch_factor=prefetch_factor)

if data_set_name == 'TINY_IMAGENET':
    from utils.dataset_handler import TinyImageNetDataset
    
    train_dir = './data/tiny-imagenet-200/'
    val_dir = './data/tiny-imagenet-200/'

    # Create Tiny ImageNet datasets
    train_dataset = TinyImageNetDataset(root_dir=train_dir, transform=transform_map[data_set_name], split='train')
    val_dataset = TinyImageNetDataset(root_dir=val_dir, transform=transform_map[data_set_name], split='val')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=number_of_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=number_of_workers, prefetch_factor=prefetch_factor)

if data_set_name == 'MINI_IMAGENET':
    from utils.dataset_handler import MiniImageNetDataset

    train_dir = './data/Mini_Imagenet/'
    val_dir = './data/Mini_Imagenet/'

    train_dataset = MiniImageNetDataset(root_dir=train_dir, transform=transform_map[data_set_name], split='train')
    val_dataset = MiniImageNetDataset(root_dir=val_dir, transform=transform_map[data_set_name], split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=number_of_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=number_of_workers, prefetch_factor=prefetch_factor)



 
# utils.show_sample_images(train_loader,num_images=32)

# import pdb
# pdb.set_trace()

    # def calculate_mean_std(loader):
    #     mean = 0.0
    #     std = 0.0
    #     total_images_count = 0
    #     for images, _ in loader:
    #         batch_images_count = images.size(0)  # number of images in the batch
    #         images = images.view(batch_images_count, images.size(1), -1)  # flatten H, W into a single dimension
    #         mean += images.mean(2).sum(0)
    #         std += images.std(2).sum(0)
    #         total_images_count += batch_images_count

    #     mean /= total_images_count
    #     std /= total_images_count
    #     return mean, std
    
    # transform = transforms.Compose([transforms.ToTensor()])
    # dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)
    # mean, std = calculate_mean_std(loader)
    # print(f"Mean: {mean}")
    # print(f"Standard Deviation: {std}")
    # import pdb
    # pdb.set_trace()
###############################################################
###############################################################



encoder = model_map[data_set_name]
expander = models.Expander(input_dim=latent_dim, output_dim=expander_dim,hidden_dim=EXPANDER_HIDDEN_DIM).to(device)
# classifier = Models.StrongerMLPClassifier(input_dim=latent_dim,num_classes=DATASET_NUMBER_OF_CLASS).to(device)
classifier = models.LinearClassifier(input_dim=latent_dim,num_classes=DATASET_NUMBER_OF_CLASS).to(device)




# Optimizer for encoder training
if optimizer == 'Adam':
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer  == 'LARS':
    # Initialize the optimizer using LARS
    encoder_optimizer = evaluation_utils.LARS(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eta=0.001,  # LARS-specific parameter, can be tuned if needed
        weight_decay_filter=lambda p: p.ndim != 1,  # Avoid weight decay on biases and batch norms
        lars_adaptation_filter=lambda p: p.ndim != 1
    )



scheduler = CosineAnnealingLR(encoder_optimizer, T_max=num_epochs)

# evaluation_utils.visualize_latent_space_with_pca_and_tsne(encoder=encoder, data_loader=train_loader, device=device, save_path=os.path.join(f"./{LOG_PATH}/embedding/", f'{-1}.png'), n_batches_to_use=100)

# import pdb; pdb.set_trace()

if not LOAD_THE_MODEL:
    encoder.train()
    loss_epochs, h_lengths, z_lengths = [], [], []
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        sampler.set_epoch(epoch)
        
        # print(f'Epoch {epoch+1}/{num_epochs}')
        # loss_epoch, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander = training_scripts.CDSSL_train(encoder, expander, encoder_optimizer, train_loader, device, transform_map[data_set_name], epoch)
        
        # loss_epoch, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander = training_scripts.train(encoder, expander, encoder_optimizer, train_loader, device, transform_map[data_set_name], epoch)

        # loss_epoch, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander = dist.spawn(training_scripts.CDSSL_train_distributed, args=(world_size,encoder, expander, encoder_optimizer, train_loader, device, transform_map[data_set_name], epoch,), nprocs=world_size)
        loss_epoch, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander = training_scripts.CDSSL_train_distributed(
             world_size, dist.get_rank(), encoder, expander, encoder_optimizer, train_loader, device, transform_map[data_set_name], epoch
        )

        # loss_epoch, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander = \
        #                     training_scripts.train_with_smooth_weights(encoder, expander, encoder_optimizer, \
        #                                 train_loader, device, transform_map[data_set_name], epoch,T=num_epochs,beta=np.log(num_epochs)/num_epochs) 


        loss_epochs.append(loss_epoch)
        h_lengths.append(avg_embedding_lengths_before_expander)
        z_lengths.append(avg_embedding_lengths_after_expander)
        scheduler.step()

        # import pdb; pdb.set_trace()

        evaluation_utils.plot_loss_values(loss_epochs=loss_epochs, save_path=os.path.join(f"logs/{LOG_PATH}/loss/", f'{epoch}.png'))
        evaluation_utils.plot_embedding_length(embedding_lengths=h_lengths, save_path=os.path.join(f"logs/{LOG_PATH}/h_length/", f'{epoch}.png'))
        evaluation_utils.plot_embedding_length(embedding_lengths=z_lengths, save_path=os.path.join(f"logs/{LOG_PATH}/z_length/", f'{epoch}.png'))
        # evaluation_utils.visualize_latent_space_with_pca_and_tsne(encoder=encoder, data_loader=train_loader, device=device, save_path=os.path.join(f"LOGS/{LOG_PATH}/embedding/", f'{epoch}.png'), n_batches_to_use=100)
        if epoch%100 == 0:
            evaluation_utils.visualize_latent_space_umap(encoder=encoder, data_loader=train_loader, device=device, save_path=os.path.join(f"logs/{LOG_PATH}/embedding/", f'{epoch}.png'), n_batches_to_use=100)
    

# Evaluate the learned representations using a linear classifier
print("Training classifier on frozen encoder representations...")

if SAVE_THE_MODEL:
    evaluation_utils.save_model(encoder)

if LOAD_THE_MODEL:
    evaluation_utils.load_model(encoder, MODEL_PATH)


with evaluation_utils.redirect_terminal_output(os.path.join(f"logs/{LOG_PATH}/classifier_results/", "classifier.txt")):
    if test_using_classifier:
        encoder.eval()  # Freeze the encoder weights

        # Optimizer for classifier training
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        # Train the classifier using labeled data

        # if data_set_name == 'MINI_IMAGENET': 
        #     XFORM = transforms.Compose([
        #         transforms.Resize((84,84)),
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #     ])

        #     train_dataset = MiniImageNetDataset(root_dir=train_dir, transform=XFORM, split='train')
        #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        #                             num_workers=number_of_workers, prefetch_factor=prefetch_factor)
        training_scripts.train_classifier(encoder, classifier, classifier_optimizer, train_loader, device)
        training_scripts.test_classifier(encoder,classifier,test_loader,device)
        

    # Visualize the learned latent space using PCA and t-SNE
    # print("Visualizing latent space with PCA and t-SNE...")
    # evaluation_utils.visualize_latent_space_with_tsne(encoder, test_loader, device)

    # evaluation_utils.visualize_latent_space_with_pca_and_tsne(encoder, test_loader, device)

    # Evaluate learned representations using k-NN after training
    # print("Evaluating learned representations using k-NN classifier...")
    # knn_accuracy, ari, nmi = evaluation_utils.evaluate_knn(encoder, train_loader, test_loader, k=number_of_clusters, device=device)

    