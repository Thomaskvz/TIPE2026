import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, nb_entree, nb_cache, nb_sortie):
        super().__init__()
        self.linear1 = nn.Linear(nb_entree, nb_cache)
        self.linear2 = nn.Linear(nb_cache, nb_sortie)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, nom="model_dqn.pth"):
        dossierModels = "./models"
        nomFichier = os.path.join(dossierModels, nom)
        torch.save(self.state_dict(), nomFichier) # Ne sauvegarde que les poids

    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss # Mean Squared Error ou Erreur quadratique moyenne

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0) # rajoute une dimension
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done,)

        # Valeur de Q prédite par l'IA dans l'état actuel
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                # Q_new = r + \gamma * max(prochaine valeur de Q prédite)
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()
        