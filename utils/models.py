import torch.nn as nn
import torch
from torchvision.models import resnet18

from torchvision.models import resnet50





class Projector(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )
            
    def forward(self, x):
        return self.network(x)
    
class encoder_expander(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
 
        self.encoder = resnet18(num_classes=encoder_dim)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=2)
        self.encoder.maxpool = nn.Identity()
        
        self.projector = Projector(encoder_dim, projector_dim)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2))
        y = self.encoder(x)
        return self.projector(y).chunk(2)



# class ResNetEncoder50(nn.Module):
#     def __init__(self, latent_dim):
#         super(ResNetEncoder50, self).__init__()
#         resnet = resnet50(pretrained=False)
#         resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Modify for CIFAR
#         resnet.maxpool = nn.Identity()  # Remove max-pooling for CIFAR
#         self.features = nn.Sequential(*list(resnet.children())[:-2])  # Exclude FC layers
#         self.fc = None  # Initialize fully connected later
#         self.latent_dim = latent_dim

#     def forward(self, x):
#         x = self.features(x)  # Extract features
#         x = torch.flatten(x, 1)  # Flatten the spatial dimensions
#         if self.fc is None:
#             self.fc = nn.Linear(x.size(1), self.latent_dim).to(x.device)  # Infer input size
#         x = self.fc(x)  # Map to latent space
#         return x


 
# class EnhancedEncoderCifar(nn.Module):
#     def __init__(self, latent_dim):
#         super(EnhancedEncoderCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.fc = nn.Linear(128 * 8 * 8, latent_dim)

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.relu(self.bn3(self.conv3(x)))
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc(x)
#         return x


# class ResNetEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(ResNetEncoder, self).__init__()
#         resnet = resnet18(num_class=latent_dim)
#         resnet.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),stride=1)
#         resnet.maxpool = nn.Identity()


#         resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
#         resnet.maxpool = nn.Identity()  # Remove max-pooling for CIFAR
#         self.features = nn.Sequential(*list(resnet.children())[:-2])  # Exclude FC layers
#         self.fc = None  # Initialize fully connected later
#         self.latent_dim = latent_dim

#     def forward(self, x):
#         x = self.features(x)  # Extract features
#         x = torch.flatten(x, 1)  # Flatten the spatial dimensions
#         if self.fc is None:
#             self.fc = nn.Linear(x.size(1), self.latent_dim).to(x.device)  # Infer input size
#         x = self.fc(x)  # Map to latent space
#         return x




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = out + self.shortcut(x)
        out = out.clone() + self.shortcut(x)
        out = torch.relu(out)
        return out

class EnhancedEncoderMiniImageNetResidual(nn.Module):
    def __init__(self, latent_dim=512):
        super(EnhancedEncoderMiniImageNetResidual, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stacked residual blocks
        self.layer1 = self._make_layer(ResidualBlock, 64, 128, stride=2)   # (128, 42, 42)
        self.layer2 = self._make_layer(ResidualBlock, 128, 256, stride=2)  # (256, 21, 21)
        self.layer3 = self._make_layer(ResidualBlock, 256, 512, stride=2)  # (512, 11, 11)
        self.layer4 = self._make_layer(ResidualBlock, 512, 1024, stride=2) # (1024, 6, 6)
        
        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer to map to latent dimension
        self.fc = nn.Linear(1024, latent_dim)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = [block(in_channels, out_channels, stride)]
        layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))    # Initial conv layer
        x = self.layer1(x)                         # Residual Block Layer 1
        x = self.layer2(x)                         # Residual Block Layer 2
        x = self.layer3(x)                         # Residual Block Layer 3
        x = self.layer4(x)                         # Residual Block Layer 4
        
        x = self.avgpool(x)                        # Pooling
        x = torch.flatten(x, 1)                    # Flatten
        x = self.fc(x)                             # Map to latent dimension
        
        return x



class EncoderMiniImageNet(nn.Module):
    def __init__(self, latent_dim=512):
        super(EncoderMiniImageNet, self).__init__()
        
        # Define the convolutional layers with BatchNorm for better stability
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # A final pooling layer to reduce the spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # A fully connected layer to map to the desired latent dimension
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))    # (64, 84, 84)
        x = torch.relu(self.bn2(self.conv2(x)))    # (128, 42, 42)
        x = torch.relu(self.bn3(self.conv3(x)))    # (256, 21, 21)
        x = torch.relu(self.bn4(self.conv4(x)))    # (512, 11, 11)
        x = torch.relu(self.bn5(self.conv5(x)))    # (1024, 6, 6)
        
        x = self.avgpool(x)                        # (1024, 1, 1)
        x = torch.flatten(x, 1)                    # (1024)
        x = self.fc(x)                             # Map to latent_dim
        
        return x


 
 
 
class Encoder_stsl(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_stsl, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Change input channels to 3 for RGB
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 24 * 24, latent_dim)  # Adjust to 32 * 24 * 24 for STL-10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class Encoder_cifar(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, latent_dim) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes=10):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

 
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)  
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  




class Bigger_Expander(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Expander, self).__init__()
        self.expander = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim),        
            nn.ReLU(),                         
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),        
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, x):
        return self.expander(x)
    
# class Expander_little(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=128): 
#         super(Expander_little, self).__init__()
#         self.expander = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),  
#             nn.BatchNorm1d(hidden_dim),        
#             nn.ReLU(),                         
#             nn.Linear(hidden_dim, output_dim)  
#         )

#     def forward(self, x):
#         return self.expander(x)
    



class Expander(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=8192):
        super(Expander, self).__init__()
        self.expander = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        return self.expander(x)


###############


import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomFeatureExtractor(nn.Module):
    def __init__(self, latent_dim=128):
        super(CustomFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (Batch, 64, 42, 42)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (Batch, 128, 21, 21)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (Batch, 256, 10, 10)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: (Batch, 512, 5, 5)
        )

        # Adaptive Pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (Batch, 512, 1, 1)

        # Fully connected layer for latent dimension projection
        self.fc = nn.Sequential(
            nn.Linear(512, latent_dim),  # Input 512 -> Latent 128
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)             # Convolutional feature extraction
        x = self.adaptive_pool(x)       # Ensure consistent output size
        x = torch.flatten(x, 1)         # Flatten to (Batch, 512)
        x = self.fc(x)                  # Project to latent dimension
        return x



 


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class StrongerMLPClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes=100):
#         super(StrongerMLPClassifier, self).__init__()
#         # First layer
#         self.fc1 = nn.Linear(input_dim, 512)  # Increased hidden dimensions
#         self.bn1 = nn.BatchNorm1d(512)
        
#         # Second layer
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
        
#         # Third layer
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
        
#         # Final classification layer
#         self.fc4 = nn.Linear(128, num_classes)
        
#         # Regularization
#         self.dropout = nn.Dropout(0.4)  # Increased dropout for better regularization

#     def forward(self, x):
#         # First layer with activation and batch normalization
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
        
#         # Second layer
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
        
#         # Third layer
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = self.dropout(x)
        
#         # Final classification layer (logits)
#         x = self.fc4(x)
#         return x


 

class StrongerMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=100, hidden_dims=[1024, 512, 256]):
        super(StrongerMLPClassifier, self).__init__()
        
        # Dynamically construct hidden layers with batch normalization
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Stabilizes training
            layers.append(nn.ReLU())      # Efficient activation
            layers.append(nn.Dropout(0.05))            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

 

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)
