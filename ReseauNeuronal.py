import numpy as np

class MLP:

    def initialisation(self, nbr_neurones , activation ='relu', T=0.01):

        print("liste du nombre de neurones par couche:")
        self.nbr_neurones = nbr_neurones
        print(self.nbr_neurones)

        print("la fonction d'activation est : ")
        self.activation = self.relu
        print(self.activation.__name__)

        print("la dérivée de la fonction d'activation est :")
        self.activation_dérivée =self.relu_dérivée
        print(self.activation_dérivée.__name__)

        print("le taux d'apprentissage est ")
        self.T = T
        print(self.T)
        
        print("la liste de matrices de poids aléatoires")
        self.poids = [np.random.randn(nbr_neurones[i + 1],nbr_neurones[i])* np.sqrt(2 / nbr_neurones[i]) for i in range(len(nbr_neurones) - 1)]
        for i, W in enumerate(self.poids):
            print(f"Poids couche {i+1} :\n{W}\n")

        print("la liste de vecteurs biais initialisés à 0 :")
        self.biases = [np.zeros((nbr_neurones[i + 1], 1)) for i in range(len(nbr_neurones) - 1)]
        for i, b in enumerate(self.biases):
            print(f"Biais couche {i+1} :\n{b}\n")


    def relu(self, Z):
      return np.maximum(0, Z)
    
    def relu_dérivée(self, Z):
      return (Z > 0).astype(float)
    
    
    def propagation_avant(self,X):
       A = X.T

       print("la liste des activations de chaque couche :")
       activations = [A]
       print(f"Activation entrée  :\n{A.T}\n")

       print(" la liste avant activation")
       Avant_act = []

       for i in range(len(self.poids)):
            
            Z = np.dot(self.poids[i], A) + self.biases[i]
            Avant_act.append(Z)
            print(f"Z couche {i+1} :\n{Z.T}\n")

            if i == len(self.poids) - 1:  
                A = self.sigmoid(Z)  
            else:
                A = self.activation(Z)
            activations.append(A)

            print(f"Activation après couche {i+1} :\n{A.T}\n")
            
       return A, activations, Avant_act
    

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    


def main():     
    
    fichier_test = "données.txt"  
    données = np.loadtxt(fichier_test)

    print("Données d'entrée :")
    D = données[:, :-1]  
    print(D)

    print("Sorties attendues :")
    dernière_colonne = données[:, -1] 
    print(dernière_colonne)

    test = MLP()
    test.initialisation([3, 5, 3, 1], activation='relu', T=0.01)

    A, activations, Avant_act=test.propagation_avant(D)
    print("sortie du réseau : ")
    print(A.T)

if __name__ == "__main__":
    main()