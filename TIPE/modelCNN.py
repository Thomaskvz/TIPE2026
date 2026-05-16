import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Conv_QNet(nn.Module):
    def __init__(self, num_actions=3):
        super().__init__()
        
        # Entrée : [Batch, 1 canal (Gris), 36 Hauteur, 80 Largeur]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Taille -> 18 x 40
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Taille -> 9 x 20
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Taille -> 4 x 10 (9//2 = 4)

        # Taille après aplatissement (Flatten) : 64 filtres * 4 * 10 = 2560
        self.fc1 = nn.Linear(64 * 4 * 10, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # Passages dans les couches Convolutives + Activations + Pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Aplatissement dynamique en préservant la dimension de Batch
        x = x.view(x.size(0), -1) 
        
        # Couches fully-connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Si on a un seul échantillon (Short Memory), on ajoute la dimension de Batch
        if len(state.shape) == 3: # Forme: (1, H, W) -> devient (1, 1, H, W)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: Prédictions des valeurs Q actuelles
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))

            target[idx][action[idx].item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()