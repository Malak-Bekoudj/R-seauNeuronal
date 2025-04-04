import numpy as np
from sklearn.model_selection import train_test_split       
#Data preparation
print("\n=== Étape 1: Préparer les données ===")

donnees =np.loadtxt("data.txt")
points_3D=donnees[:, :3]
etiquettes=donnees[:, 3]   # 1 = vase, 0 = bruit

points_3D=(points_3D-points_3D.mean(axis=0))/points_3D.std(axis=0)
X_train,X_test,y_train,y_test=train_test_split(points_3D,etiquettes,test_size=0.3)

print(f"Nombre total de points : {len(points_3D)}")
print(f"Points vase : {int(etiquettes.sum())}")
print("\nExemple de point normalisé :")
print(f"Coordonnées : {X_train[0]}")
print(f"Étiquette : {y_train[0]}")    

#Creation of the Neural Network
print("\n=== Étape 2: Créer le réseau de neurones ===")

class MLP:
    def __init__(self, couches, activation='relu'):
        self.couches = couches
        self.activation = activation

        self.poids = []
        self.biais = []
        for i in range(len(couches)-1):
            taille_couche=couches[i]

            if activation=='relu':
                self.poids.append(np.random.randn(couches[i+1], taille_couche) * np.sqrt(2./taille_couche))
            else:  # tanh
                self.poids.append(np.random.randn(couches[i+1], taille_couche) * np.sqrt(1./taille_couche))
            self.biais.append(np.zeros((couches[i+1], 1)))



    def fonction_activation(self,x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return np.tanh(x)
        

    def fonction_sigmoide(self, x):
        return 1/(1+np.exp(-x)) 

    def propagation_avant(self, X):
        activation = X.T
        for poids, biais in zip(self.poids[-1], self.biais[:-1]):
            z = np.dot(poids, activation )+ biais
            activation = self.fonction_activation(z)

        #Dernière couche 
        sortie = np.dot(self.poids[-1],activation)+ self.biais[-1]
        return self.fonction_sigmoide(sortie)      