import numpy as np
from sklearn.model_selection import train_test_split       
from sklearn.metrics import classification_report, confusion_matrix
#Data preparation
print("\n===  Préparer les données ===")

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
print("\n=== Créer le réseau de neurones ===")

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
        for poids, biais in zip(self.poids[:-1], self.biais[:-1]):
            z = np.dot(poids, activation )+ biais
            activation = self.fonction_activation(z)

        #Dernière couche 
        sortie = np.dot(self.poids[-1],activation)+ self.biais[-1]
        return self.fonction_sigmoide(sortie)      

    def entrainement(self,X,y,iterations=1000,taux_apprentissage=0.1):
        y = y.reshape(1,-1)

        for iteration in range(iterations):
            #propagation avant
            activations = [X.T]
            valeurs_z = []

            for poids, biais in zip(self.poids[:-1], self.biais[:-1]):
                z= np.dot(poids,activations[-1])+ biais
                a = self.fonction_activation(z)
                valeurs_z.append(z)
                activations.append(a)
            sortie = self.fonction_sigmoide(np.dot(self.poids[-1],activations[-1])+self.biais[-1])

            # Rétropropagation
            erreur = sortie - y 
            for i in reversed(range(len(self.poids))):
                #le gradient
                gradient = erreur @ activations[i].T / X.shape[0]
                #mise a jour
                self.poids[i] -=taux_apprentissage * gradient
                self.biais[i] -= taux_apprentissage * erreur.mean(axis=1, keepdims=True)            

                if i > 0:
                    erreur = (self.poids[i].T @ erreur) * (valeurs_z[i-1] > 0 if self.activation == 'relu' else (1 - np.tanh(valeurs_z[i-1])**2))

            if iteration % 100 == 0:
                perte = -np.mean(y * np.log(sortie+1e-8) +(1-y) * np.log(1-sortie+1e-8)) 
                print(f"Itération {iteration}: Perte = {perte:.4f}")


# XOR 
print("\n=== Tester avec XOR ===")
A=np.array([[0,0],[0,1],[1,0],[1,1]])
B=np.array([0, 1, 1, 0])

reseau=MLP([2,4,1], activation='tanh')
reseau.entrainement(A,B,iterations=10000,taux_apprentissage=0.1)

print("\nRésultats pour XOR:")
for x, y in zip(A, B):
    prediction = reseau.propagation_avant(x.reshape(1,-1))
    print(f"Entrée {x} → Sortie:{1 if prediction > 0.5 else 0} (Attendu: {y})")


# Entraînement sur les données du vase
print("\n\n=== Entraîner sur les vases ===")
vase=MLP([3,32,16,1], activation='relu')
vase.entrainement(X_train,y_train,iterations=10000,taux_apprentissage=0.1)

#Évaluation et amélioration
print("\n=== Évaluer et améliorer ===")
def evaluation(modele,X,y):
    predictions=np.array([modele.propagation_avant(x.reshape(1,-1)) > 0.5 for x in X]).flatten()

    print("\nRapport de classification:")
    print(classification_report(y, predictions))

    print("\nMatrice de confusion:")
    print(confusion_matrix(y, predictions))

    precision=np.mean(predictions == y)
    print(f"\nPrécision globale: {precision*100:.2f}%")
    return precision
precision=evaluation(vase,X_test,y_test)

if precision < 0.9 :
    print("\nAmélioration du réseau ")
    vase=MLP([3,64,32,1], activation='relu')
    vase.entrainement(X_train,y_train,iterations=2000,taux_apprentissage=0.1)
    evaluation(vase,X_test,y_test)

