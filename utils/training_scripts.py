from tqdm import tqdm
import torch 
import torch.nn as nn
import yaml, os
from torchvision import datasets, transforms
import utils.custom_loss as custom_loss
import geomloss 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import utils
import numpy as np
import utils.evaluation_utils
from utils import evaluation_utils
import utils.custom_loss_CDSSL as custom_loss_CDSSL
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def init_process_group():
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())
    




with open("/home/hadi/Desktop/WorkSpace/SSL/LMA-OT/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# batch_size = config['hyperparameters']['batch_size']
# learning_rate = config['hyperparameters']['learning_rate']
# num_epochs = config['hyperparameters']['num_epochs']
classifier_epochs = config['hyperparameters']['classifier_epochs']
gradient_clipping = config['hyperparameters']['gradient_clipping']
VARIANCE_THRESH = config['hyperparameters']['VARIANCE_THRESH']
VARIANCE_eps = config['hyperparameters']['VARIANCE_eps']

# latent_dim = config['hyperparameters']['latent_dim']
kernel_vicreg_weight = config['loss_weights']['kernel_vicreg']
kernel_vicreg_parts_weight = config['loss_weights']['kernel_vicreg_parts']
hsic_weight = config['loss_weights']['hsic']
jsd_weight = config['loss_weights']['jsd']
ot_weight = config['loss_weights']['ot_loss']
mmd_weight = config['loss_weights']['mmd']
var_weight = config['loss_weights']['var_loss']
cov_weight = config['loss_weights']['cov_loss']
ent_weight = config['loss_weights']['ent_loss']
inv_weight = config['loss_weights']['inv_loss']
lambda_barlow = config['loss_weights']['lambda_barlow']
auto_cor_weight = config['loss_weights']['auto_cor_weight']

cross_correlation_between_samples_weight = config['loss_weights']['cross_correlation_between_samples_weight']
cross_correlation_between_features_weight = config['loss_weights']['cross_correlation_between_features_weight']
auto_correlation_between_samples_weight = config['loss_weights']['auto_correlation_between_samples_weight']
auto_correlation_between_features_weight = config['loss_weights']['auto_correlation_between_features_weight']
cross_dependence_between_samples_weight = config['loss_weights']['cross_dependence_between_samples_weight']
cross_dependence_between_features_weight = config['loss_weights']['cross_dependence_between_features_weight']
auto_dependence_between_samples_weight = config['loss_weights']['auto_dependence_between_samples_weight']
auto_dependence_between_features_weight = config['loss_weights']['auto_dependence_between_features_weight']

def CDSSL_train(model, optimizer, data_loader, device, transform, epoch ,expander):

    total_loss = 0

    # Transformation to convert Tensor to PIL Image
    to_pil_image = transforms.ToPILImage()
    # scaler = torch.amp.GradScaler('cuda')

    embedding_lengths_before_expander, embedding_lengths_after_expander = [], []
    # Use tqdm for progress bar
    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for images, _ in tepoch:
            tepoch.set_description(f"Training")

            optimizer.zero_grad()

            view1 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            view2 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            


            # evaluation_utils.visualize_views(view1, view2)
            # with torch.amp.autocast('cuda'):
            # h1 = model(view1)
            # h2 = model(view2)
            # h1 = F.normalize(embed1, p=2, dim=1)
            # h2 = F.normalize(embed2, p=2, dim=1)
            DS = "!MNIST"

            if DS != "MNIST":
                z1, z2  = model(view1,view2)
            else:
                h1 = model(view1)
                h2 = model(view2)
                z1 = expander(h1)
                z2 = expander(h2)

            # embedding_lengths_before_expander.append((torch.mean(torch.linalg.norm(h1, axis=1)).item() + torch.mean(torch.linalg.norm(h2, axis=1)).item()) * 0.5)
            # z1 = h1
            # z2 = h2

            # z1 = expander(h1)
            # z2 = expander(h2)

            embedding_lengths_after_expander.append((torch.mean(torch.linalg.norm(z1, axis=1)).item() + torch.mean(torch.linalg.norm(z2, axis=1)).item()) * 0.5)

            # print(view1.device, view2.device, h1.device, h2.device, z1.device, z2.device)

            cross_correlation_between_samples_loss = custom_loss_CDSSL.cross_correlation_between_samples(z1, z2, lambda_param=0.05)
            cross_correlation_between_features_loss = custom_loss_CDSSL.cross_correlation_between_features(z1, z2, lambda_param=0.05)
            auto_correlation_between_samples_loss = custom_loss_CDSSL.auto_correlation_between_samples(z1, z2)
            auto_correlation_between_features_loss = custom_loss_CDSSL.auto_correlation_between_features(z1, z2)
            cross_dependence_between_samples_loss = custom_loss_CDSSL.cross_dependence_between_samples(z1, z2, method="HSIC")
            cross_dependence_between_features_loss = custom_loss_CDSSL.cross_dependence_between_features(z1, z2, method="HSIC")
            auto_dependence_between_samples_loss = custom_loss_CDSSL.auto_dependence_between_samples(z1, z2)
            auto_dependence_between_features_loss = custom_loss_CDSSL.auto_dependence_between_features(z1, z2)


            # import pdb; pdb.set_trace()

            loss = (
            cross_correlation_between_samples_weight * cross_correlation_between_samples_loss
            + cross_correlation_between_features_weight * cross_correlation_between_features_loss
            + auto_correlation_between_samples_weight * auto_correlation_between_samples_loss
            + auto_correlation_between_features_weight * auto_correlation_between_features_loss
            + cross_dependence_between_samples_weight * cross_dependence_between_samples_loss
            + cross_dependence_between_features_weight * cross_dependence_between_features_loss
            + auto_dependence_between_samples_weight * auto_dependence_between_samples_loss
            + auto_dependence_between_features_weight * auto_dependence_between_features_loss
            )


            cross_correlation_between_samples = cross_correlation_between_samples_weight * cross_correlation_between_samples_loss
            cross_correlation_between_features = cross_correlation_between_features_weight * cross_correlation_between_features_loss
            auto_correlation_between_samples = auto_correlation_between_samples_weight * auto_correlation_between_samples_loss
            auto_correlation_between_features = auto_correlation_between_features_weight * auto_correlation_between_features_loss
            cross_dependence_between_samples = cross_dependence_between_samples_weight * cross_dependence_between_samples_loss
            cross_dependence_between_features = cross_dependence_between_features_weight * cross_dependence_between_features_loss
            auto_dependence_between_samples = auto_dependence_between_samples_weight * auto_dependence_between_samples_loss
            auto_dependence_between_features = auto_dependence_between_features_weight * auto_dependence_between_features_loss

            # print ('here')
            # import pdb
            # pdb.set_trace()
            # loss = var_weight * var_loss + cov_weight * cov_loss + inv_weight * inv_loss + jsd_weight * jsd
            # sparsity = Custom_loss.sparse_orthogonalization_loss(z1) #+ Custom_loss.sparse_orthogonalization_loss(z2)
            # sparsity = Custom_loss.sparse_orthogonalization_lasso_loss (z1, lambda_lasso=0.01) #+ Custom_loss.sparse_orthogonalization_lasso_loss(z2,  lambda_lasso=0.01)
            # loss = loss + 0.05 * sparsity
            


            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer.step()
            total_loss = total_loss +  loss.item()
            tepoch.set_postfix( 
                
            A = cross_correlation_between_samples.item(),
            B = cross_correlation_between_features.item(),
            C = auto_correlation_between_samples.item(),
            D = auto_correlation_between_features.item(),
            E = cross_dependence_between_samples.item(),
            F = cross_dependence_between_features.item(),
            G = auto_dependence_between_samples.item(),
            H = auto_dependence_between_features.item()
            )

    avg_loss = total_loss / len(data_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')

    avg_embedding_lengths_before_expander = np.mean(embedding_lengths_before_expander)
    avg_embedding_lengths_after_expander = np.mean(embedding_lengths_after_expander)

    # utils.plot_embedding_lengths(h1.cpu().detach().numpy(),save_path=os.path.join("./log/embedding_length/", f'{epoch}.png'))

    return avg_loss, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander





def CDSSL_train_distributed(world, rank, model, expander, optimizer, data_loader, device, transform, epoch):
    

    torch.cuda.set_device(rank) 
    model.to(rank)
    expander.to(rank)
    model = DDP(model, device_ids=[rank])
    expander = DDP(expander, device_ids=[rank])

    total_loss = 0

    # Transformation to convert Tensor to PIL Image
    to_pil_image = transforms.ToPILImage()
    # scaler = torch.amp.GradScaler('cuda')

    embedding_lengths_before_expander, embedding_lengths_after_expander = [], []
    # Use tqdm for progress bar
    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for images, _ in tepoch:
            tepoch.set_description(f"Training")

            optimizer.zero_grad()

            view1 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            view2 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            
            # evaluation_utils.visualize_views(view1, view2)
            # with torch.amp.autocast('cuda'):
            h1 = model(view1)
            h2 = model(view2)
            # h1 = F.normalize(embed1, p=2, dim=1)
            # h2 = F.normalize(embed2, p=2, dim=1)

            embedding_lengths_before_expander.append((torch.mean(torch.linalg.norm(h1, axis=1)).item() + torch.mean(torch.linalg.norm(h2, axis=1)).item()) * 0.5)
            # z1 = h1
            # z2 = h2

            z1 = expander(h1)
            z2 = expander(h2)

            embedding_lengths_after_expander.append((torch.mean(torch.linalg.norm(z1, axis=1)).item() + torch.mean(torch.linalg.norm(z2, axis=1)).item()) * 0.5)

            # print(view1.device, view2.device, h1.device, h2.device, z1.device, z2.device)

            cross_correlation_between_samples_loss = custom_loss_CDSSL.cross_correlation_between_samples(z1, z2, lambda_param=0.05)
            cross_correlation_between_features_loss = custom_loss_CDSSL.cross_correlation_between_features(z1, z2, lambda_param=0.05)
            auto_correlation_between_samples_loss = custom_loss_CDSSL.auto_correlation_between_samples(z1, z2)
            auto_correlation_between_features_loss = custom_loss_CDSSL.auto_correlation_between_features(z1, z2)
            cross_dependence_between_samples_loss = custom_loss_CDSSL.cross_dependence_between_samples(z1, z2, method="HSIC")
            cross_dependence_between_features_loss = custom_loss_CDSSL.cross_dependence_between_features(z1, z2, method="HSIC")
            auto_dependence_between_samples_loss = custom_loss_CDSSL.auto_dependence_between_samples(z1, z2)
            auto_dependence_between_features_loss = custom_loss_CDSSL.auto_dependence_between_features(z1, z2)


            # import pdb; pdb.set_trace()

            loss = (
            cross_correlation_between_samples_weight * cross_correlation_between_samples_loss
            + cross_correlation_between_features_weight * cross_correlation_between_features_loss
            + auto_correlation_between_samples_weight * auto_correlation_between_samples_loss
            + auto_correlation_between_features_weight * auto_correlation_between_features_loss
            + cross_dependence_between_samples_weight * cross_dependence_between_samples_loss
            + cross_dependence_between_features_weight * cross_dependence_between_features_loss
            + auto_dependence_between_samples_weight * auto_dependence_between_samples_loss
            + auto_dependence_between_features_weight * auto_dependence_between_features_loss
            )


            cross_correlation_between_samples = cross_correlation_between_samples_weight * cross_correlation_between_samples_loss
            cross_correlation_between_features = cross_correlation_between_features_weight * cross_correlation_between_features_loss
            auto_correlation_between_samples = auto_correlation_between_samples_weight * auto_correlation_between_samples_loss
            auto_correlation_between_features = auto_correlation_between_features_weight * auto_correlation_between_features_loss
            cross_dependence_between_samples = cross_dependence_between_samples_weight * cross_dependence_between_samples_loss
            cross_dependence_between_features = cross_dependence_between_features_weight * cross_dependence_between_features_loss
            auto_dependence_between_samples = auto_dependence_between_samples_weight * auto_dependence_between_samples_loss
            auto_dependence_between_features = auto_dependence_between_features_weight * auto_dependence_between_features_loss

            # print ('here')
            # import pdb
            # pdb.set_trace()
            # loss = var_weight * var_loss + cov_weight * cov_loss + inv_weight * inv_loss + jsd_weight * jsd
            # sparsity = Custom_loss.sparse_orthogonalization_loss(z1) #+ Custom_loss.sparse_orthogonalization_loss(z2)
            # sparsity = Custom_loss.sparse_orthogonalization_lasso_loss (z1, lambda_lasso=0.01) #+ Custom_loss.sparse_orthogonalization_lasso_loss(z2,  lambda_lasso=0.01)
            # loss = loss + 0.05 * sparsity
            


            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer.step()
            total_loss = total_loss +  loss.item()
            tepoch.set_postfix( 
                
            A = cross_correlation_between_samples.item(),
            B = cross_correlation_between_features.item(),
            C = auto_correlation_between_samples.item(),
            D = auto_correlation_between_features.item(),
            E = cross_dependence_between_samples.item(),
            F = cross_dependence_between_features.item(),
            G = auto_dependence_between_samples.item(),
            H = auto_dependence_between_features.item()
            )

    avg_loss = total_loss / len(data_loader)
    if rank == 0:
        print(f'Average Training Loss: {avg_loss:.4f}')

    avg_embedding_lengths_before_expander = np.mean(embedding_lengths_before_expander)
    avg_embedding_lengths_after_expander = np.mean(embedding_lengths_after_expander)

    # utils.plot_embedding_lengths(h1.cpu().detach().numpy(),save_path=os.path.join("./log/embedding_length/", f'{epoch}.png'))

    return avg_loss, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander




def train_with_smooth_weights(model, optimizer, data_loader, device, transform, epoch, T, beta):
    """
    Training function with smooth weighting for three loss terms.

    Parameters:
    - model: Main model.
    - expander: Expander model.
    - optimizer: Optimizer for training.
    - data_loader: DataLoader for training data.
    - device: Device to run training on (CPU or GPU).
    - transform: Transformation to apply to images.
    - epoch: Current epoch.
    - T: Total number of epochs for training.
    - beta: Smoothness factor for weighting system.
    """
    total_loss = 0

    # Transformation to convert Tensor to PIL Image
    to_pil_image = transforms.ToPILImage()

    # Track embedding lengths
    embedding_lengths_before_expander, embedding_lengths_after_expander = [], []


    # Get the weights for the current epoch
    weights = evaluation_utils.smooth_weights(epoch, T, beta)

    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for images, _ in tepoch:
            tepoch.set_description(f"Training")

            optimizer.zero_grad()

            # Apply transformations
            view1 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            view2 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)

            # Compute embeddings
            embed1 = model(view1)
            embed2 = model(view2)

            z1 = expander(embed1)
            z2 = expander(embed2)

            # Track embedding lengths
            embedding_lengths_before_expander.append(
                (torch.mean(torch.linalg.norm(embed1, axis=1)).item() +
                 torch.mean(torch.linalg.norm(embed2, axis=1)).item()) * 0.5
            )
            embedding_lengths_after_expander.append(
                (torch.mean(torch.linalg.norm(z1, axis=1)).item() +
                 torch.mean(torch.linalg.norm(z2, axis=1)).item()) * 0.5
            )


            loss1 = 0.005 * custom_loss.coross_corolation(z1,z2,lambda_barlow)

            loss2 = 10 * custom_loss.auto_corrolation(z1) + 10 * custom_loss.auto_corrolation(z2)  

            
            loss3 =  (-100 * custom_loss.hsic_between_corresponding_batches(z1=z1, z2=z2, scramble_z2=False)) + \
                                              (10 * custom_loss.hsic_between_corresponding_batches(z1=z1.T, z2=z2.T, scramble_z2=True))  \
                                    + 1 * custom_loss.invariance_loss(z1,z2)       
            # import pdb;pdb.set_trace()
    
            loss = (
                weights[0] * loss1 +
                weights[1] * loss2 +
                weights[2] * loss3
            )

            loss.backward()
            optimizer.step()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            total_loss = total_loss +  loss.item()

    avg_loss = total_loss / len(data_loader)
    avg_embedding_lengths_before_expander = np.mean(embedding_lengths_before_expander)
    avg_embedding_lengths_after_expander = np.mean(embedding_lengths_after_expander)

    print(f'Average Training Loss: {avg_loss:.4f}')
    return avg_loss, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander


def train_kernelized(loss_fn, model, optimizer, data_loader, device, transform, epoch,expander ):

    total_loss = 0

    # Transformation to convert Tensor to PIL Image
    to_pil_image = transforms.ToPILImage()
    # scaler = torch.amp.GradScaler('cuda')

    embedding_lengths_before_expander, embedding_lengths_after_expander = [], []
    # Use tqdm for progress bar
    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for images, _ in tepoch:
            tepoch.set_description(f"Training")

            optimizer.zero_grad()

            view1 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            view2 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            
            # evaluation_utils.visualize_views(view1, view2)
            # # with torch.amp.autocast('cuda'):
            # h1 = model(view1)
            # h2 = model(view2)

            # z1 = expander(h1)
            # z2 = expander(h2)

            DS = "MNIST"

            if DS != "MNIST":
                z1, z2  = model(view1,view2)
            else:
                h1 = model(view1)
                h2 = model(view2)
                z1 = expander(h1)
                z2 = expander(h2)



            K_11 = z1 @ z1.t()
            K_22 = z2 @ z2.t()
            K_12 = z1 @ z2.t()

            inv_loss, var_loss, cov_loss = loss_fn(K_11, K_22, K_12)

            # import pdb;pdb.set_trace()
            
            loss = inv_loss + var_loss + cov_loss

            loss.backward()
            optimizer.step()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer.step()
            total_loss = total_loss +  loss.item()
            AVAR = var_weight * var_loss
            BCOV = cov_weight * cov_loss
            CINV = inv_weight * inv_loss  
            tepoch.set_postfix( COV=BCOV.item(),VAR=AVAR.item(), INV=CINV.item())

    avg_loss = total_loss / len(data_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')

    avg_embedding_lengths_before_expander = np.mean(embedding_lengths_before_expander)
    avg_embedding_lengths_after_expander = np.mean(embedding_lengths_after_expander)

    # utils.plot_embedding_lengths(h1.cpu().detach().numpy(),save_path=os.path.join("./log/embedding_length/", f'{epoch}.png'))

    return avg_loss, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander



def train(model, optimizer, data_loader, device, transform, epoch,expander ):

    total_loss = 0

    # Transformation to convert Tensor to PIL Image
    to_pil_image = transforms.ToPILImage()
    # scaler = torch.amp.GradScaler('cuda')

    embedding_lengths_before_expander, embedding_lengths_after_expander = [], []
    # Use tqdm for progress bar
    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for images, _ in tepoch:
            tepoch.set_description(f"Training")

            optimizer.zero_grad()

            view1 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            view2 = torch.stack([transform(to_pil_image(img)) for img in images]).to(device)
            
            # evaluation_utils.visualize_views(view1, view2)
            # # with torch.amp.autocast('cuda'):
            # h1 = model(view1)
            # h2 = model(view2)

            # z1 = expander(h1)
            # z2 = expander(h2)

            DS = "MNIST"

            if DS != "MNIST":
                z1, z2  = model(view1,view2)
            else:
                h1 = model(view1)
                h2 = model(view2)
                z1 = expander(h1)
                z2 = expander(h2)


            # embedding_lengths_after_expander.append((torch.mean(torch.linalg.norm(z1, axis=1)).item() + torch.mean(torch.linalg.norm(z2, axis=1)).item()) * 0.5)

            # print(view1.device, view2.device, h1.device, h2.device, z1.device, z2.device)

            # import pdb
            # pdb.set_trace()

            mmd = 0 if mmd_weight==0 else custom_loss.mmd_loss(z1, z2) 
            # if ot_weight==0:
            #     ot_loss = 0
            # else:
            #     sinkhorn_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05)
            #     ot_loss = sinkhorn_loss(z1, z2)
            
            var_loss = 0 if var_weight==0 else   custom_loss.variance_loss(z1,threshold=VARIANCE_THRESH, epsilon=VARIANCE_eps) + custom_loss.variance_loss(z2,threshold=VARIANCE_THRESH, epsilon=VARIANCE_eps)
            cov_loss = 0 if cov_weight==0 else   custom_loss.covariance_loss(z1) + custom_loss.covariance_loss(z2)
            # ent_loss = 0 if ent_weight==0 else  Custom_loss_old.entropy_loss(z1) + Custom_loss_old.entropy_loss(z2)
            # hsic = 0 if hsic_weight==0 else   Custom_loss.hsic_loss_selected_pairs_new(z1) + Custom_loss.hsic_loss_selected_pairs_new(z2)
            # jsd = 0 if jsd_weight==0 else  Custom_loss.jsd_loss(z1, z2)
            inv_loss = 0 if inv_weight==0 else   custom_loss.invariance_loss(z1,z2)

            # import pdb; pdb.set_trace()
            # hsic = 0 if hsic_weight==0 else   (-1 * Custom_loss.hsic_between_corresponding_batches(z1=z1, z2=z2))


            # hsic = 0 if hsic_weight==0 else   (-1 * Custom_loss.hsic_between_corresponding_batches(z1=z1, z2=z2)) + (1 * Custom_loss.hsic_between_corresponding_batches(z1=z1.T, z2=z2.T))

            hsic = 0 if hsic_weight==0 else   (-100 * custom_loss.hsic_between_corresponding_batches(z1=z1, z2=z2, scramble_z2=False)) + \
                                              (100 * custom_loss.hsic_between_corresponding_batches(z1=z1.T, z2=z2.T, scramble_z2=True))

            # import pdb; pdb.set_trace()

            # hsic = 0 if hsic_weight==0 else   (-1 * Custom_loss.hsic_between_corresponding_batches(z1=z1, z2=z2, scramble_z2=False))

            # hsic = 0 if hsic_weight==0 else   (1 * Custom_loss.hsic_between_corresponding_batches(z1=z1.T, z2=z2.T, scramble_z2=True))
          
            # corr = Custom_loss.correlation_loss(z1,z2)
            # scatter = 10000 * Custom_loss.scatter_loss_feature(z1,z2)
             
            # ent = 0.001 * (Custom_loss.batch_entropy_loss(z1) + Custom_loss.batch_entropy_loss(z2))

            kernel_vicreg_loss_value = 0 if kernel_vicreg_weight==0 else   custom_loss.kernel_vicreg(z1=z1, z2=z2, weight_invariance=kernel_vicreg_parts_weight['inv_loss'], weight_variance=kernel_vicreg_parts_weight['var_loss'], weight_covariance=kernel_vicreg_parts_weight['cov_loss'])

            
            barlow =  custom_loss.coross_corolation(z1,z2,lambda_param=lambda_barlow)

            # ortho_ent = 0  if custom_loss.combined_loss_ent_orthogonality(z1) + custom_loss.combined_loss_ent_orthogonality(z2)

            auto_correlation_loss_value = 0 if auto_cor_weight == 0 else custom_loss.auto_corrolation(z1) + custom_loss.auto_corrolation(z2)

            # import pdb
            # pdb.set_trace()

            loss = (
            # kernel_vicreg_weight * kernel_vicreg_loss_value +
            # hsic_weight * hsic +
            # jsd_weight * jsd +
            # ot_weight * ot_loss +
            # mmd_weight * mmd +
            var_weight * var_loss +
            cov_weight * cov_loss +
            # ent_weight * ent_loss + 
            inv_weight * inv_loss  
            # + barlow
            # + scatter
            # + ent
            # + ortho_ent
            # + auto_cor_weight * auto_correlation_loss_value
            )


            # print ('here')
            # import pdb
            # pdb.set_trace()
            # loss = var_weight * var_loss + cov_weight * cov_loss + inv_weight * inv_loss + jsd_weight * jsd
            # sparsity = Custom_loss.sparse_orthogonalization_loss(z1) #+ Custom_loss.sparse_orthogonalization_loss(z2)
            # sparsity = Custom_loss.sparse_orthogonalization_lasso_loss (z1, lambda_lasso=0.01) #+ Custom_loss.sparse_orthogonalization_lasso_loss(z2,  lambda_lasso=0.01)
            # loss = loss + 0.05 * sparsity
            


            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer.step()
            total_loss = total_loss +  loss.item()
            AVAR = var_weight * var_loss
            BCOV = cov_weight * cov_loss
            CINV = inv_weight * inv_loss  
            tepoch.set_postfix( COV=BCOV.item(),VAR=AVAR.item(), INV=CINV.item())

    avg_loss = total_loss / len(data_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')

    avg_embedding_lengths_before_expander = np.mean(embedding_lengths_before_expander)
    avg_embedding_lengths_after_expander = np.mean(embedding_lengths_after_expander)

    # utils.plot_embedding_lengths(h1.cpu().detach().numpy(),save_path=os.path.join("./log/embedding_length/", f'{epoch}.png'))

    return avg_loss, avg_embedding_lengths_before_expander, avg_embedding_lengths_after_expander

def train_classifier(encoder, classifier, optimizer, train_loader, device):
    classifier.train()   
    criterion = nn.CrossEntropyLoss()  

    for epoch in range(classifier_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Training Classifier Epoch {epoch+1}")

                images, labels = images.to(device), labels.to(device)
               

                ###############
                # import matplotlib.pyplot as plt

                # batch_size = 32       

                # images = images.permute(0, 2, 3, 1)

                # plt.figure(figsize=(12, 12))
                # for i in range(batch_size):
                #     plt.subplot(8, 4, i + 1)   
                #     plt.imshow(images[i].cpu().numpy())
                #     plt.axis('off')   

                # plt.show()
                # import pdb
                # pdb.set_trace()
                ###############


                # Get latent representations from the frozen encoder
                with torch.no_grad():
                    latent_representations = encoder(images)

                # Train classifier on top of the frozen encoder's representations
                outputs = classifier(latent_representations)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss = total_loss + loss.item()

                # Compute accuracy
                _, predicted = outputs.max(1)
                total = total +  labels.size(0)
                correct = correct + predicted.eq(labels).sum().item()

                tepoch.set_postfix(Loss=loss.item(), Accuracy=100. * correct / total)

        print(f'Classifier Training Epoch [{epoch+1}/{classifier_epochs}], Loss: {total_loss:.4f}, Accuracy: {100. * correct / total:.2f}%')


# def train_classifier(encoder, classifier, optimizer, train_loader, device, num_classes=10):
#     encoder.eval()  # Freeze the encoder weights
#     classifier.train()  # Set classifier to train mode

#     criterion = nn.BCEWithLogitsLoss()  # One-hot encoded classification loss

#     for epoch in range(classifier_epochs):
#         total_loss = 0.0
#         correct = 0
#         total = 0

#         with tqdm(train_loader, unit="batch") as tepoch:
#             for images, labels in tepoch:
#                 tepoch.set_description(f"Training Classifier Epoch {epoch+1}")

#                 images = images.to(device)
#                 labels = labels.to(device)
#                 ###############
#                 # import matplotlib.pyplot as plt

#                 # batch_size = 32       

#                 # images = images.permute(0, 2, 3, 1)

#                 # plt.figure(figsize=(12, 12))
#                 # for i in range(batch_size):
#                 #     plt.subplot(8, 4, i + 1)   
#                 #     plt.imshow(images[i].cpu().numpy())
#                 #     plt.axis('off')   

#                 # plt.show()
#                 # import pdb
#                 # pdb.set_trace()
# #                 ###############
#                 # Convert labels to one-hot encoding
#                 labels_one_hot = torch.zeros(labels.size(0), num_classes).to(device)
#                 labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

#                 # Get latent representations from the frozen encoder
#                 with torch.no_grad():
#                     latent_representations = encoder(images)

#                 # Train classifier on top of the frozen encoder's representations
#                 outputs = classifier(latent_representations)
#                 loss = criterion(outputs, labels_one_hot)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#                 # Compute accuracy
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()

#                 tepoch.set_postfix(Loss=loss.item(), Accuracy=100. * correct / total)

#         print(f'Classifier Training Epoch [{epoch+1}/{classifier_epochs}], Loss: {total_loss:.4f}, Accuracy: {100. * correct / total:.2f}%')

# def test_classifier(encoder, classifier, test_loader, device):
#     classifier.eval()  
#     criterion = nn.CrossEntropyLoss()   
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():   
#         with tqdm(test_loader, unit="batch") as tepoch:
#             for images, labels in tepoch:
#                 tepoch.set_description("Testing Classifier")

#                 images, labels = images.to(device), labels.to(device)

#                 latent_representations = encoder(images)

#                 outputs = classifier(latent_representations)
#                 loss = criterion(outputs, labels)

#                 total_loss += loss.item()

#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()

#                 tepoch.set_postfix(Loss=loss.item(), Accuracy=100. * correct / total)

#     print(f'Test Results, Loss: {total_loss:.4f}, Accuracy: {100. * correct / total:.2f}%')
    
#     return total_loss, 100. * correct / total


def test_classifier(encoder, classifier, test_loader, device):
    classifier.eval()  
    criterion = nn.CrossEntropyLoss()   
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():   
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description("Testing Classifier")

                images, labels = images.to(device), labels.to(device)

                latent_representations = encoder(images)

                outputs = classifier(latent_representations)
                loss = criterion(outputs, labels)

                total_loss = total_loss + loss.item()

                # Top-1 accuracy
                _, predicted = outputs.max(1)
                correct_top1 = correct_top1 +  predicted.eq(labels).sum().item()

                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 = correct_top5 +  sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))

                total = total +  labels.size(0)

                tepoch.set_postfix(
                    Loss=loss.item(), 
                    Top1_Acc=100. * correct_top1 / total,
                    Top5_Acc=100. * correct_top5 / total
                )

    print(f'Test Results, Loss: {total_loss:.4f}, '
          f'Top-1 Accuracy: {100. * correct_top1 / total:.2f}%, '
          f'Top-5 Accuracy: {100. * correct_top5 / total:.2f}%')
    
    return total_loss, 100. * correct_top1 / total, 100. * correct_top5 / total
