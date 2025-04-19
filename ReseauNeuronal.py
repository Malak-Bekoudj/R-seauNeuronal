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
    
    
    def propagation_avant(self,X,print_details=False):
       A = X.T

       if print_details:
        print("Activation entrée  :")
        print(A.T)

       activations = [A]
       Avant_act = []

       for i in range(len(self.poids)):
            
            Z = np.dot(self.poids[i], A) + self.biases[i]
            Avant_act.append(Z)
          

            if i == len(self.poids) - 1:  
                A = self.sigmoid(Z)  
            else:
                A = self.activation(Z)
            activations.append(A)

            if print_details:
             print(f"\nZ couche {i+1} :\n{Z.T}")
             print(f"\nActivation après couche {i+1} :\n{A.T}")
            
       return A, activations, Avant_act
    

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_dérivée(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    
    def retro_propagation(self, X, Y):
       A, activations, Avant_act = self.propagation_avant(X)
       m = X.shape[0]  
    
       deltas = []
       delta_sortie = (A - Y.T) * self.sigmoid_dérivée(Avant_act[-1])
       deltas.append(delta_sortie)
          
       for l in range(len(self.poids)-1, 0, -1):
         
         delta = np.dot(self.poids[l].T, deltas[-1]) * self.relu_dérivée(Avant_act[l-1])
         deltas.append(delta)
    
       deltas = deltas[::-1]
    
       
       for l in range(len(self.poids)):
         update_poids = np.dot(deltas[l], activations[l].T) / m
         update_biais = np.sum(deltas[l], axis=1, keepdims=True) / m
        
         self.poids[l] -= self.T * update_poids
         self.biases[l] -= self.T * update_biais

    def train(self, X, Y, epochs=1000):
      
      for epoch in range(epochs):
        A, activations, Avant_act = self.propagation_avant(X, print_details=False)
        loss = np.mean((A.T - Y)**2)

        self.retro_propagation(X, Y)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.5f}")


    

def main():     
    
    
    fichier_test = "données.txt"  
    données = np.loadtxt(fichier_test)

    print("Données d'entrée :")
    D = données[:, :-1]  
    print(D)

    
    D = (D - np.mean(D, axis=0)) / np.std(D, axis=0)
    print("Valeurs moyennes apres normalisation :", np.mean(D, axis=0))
    print("écart type apres normalisation :", np.std(D, axis=0))

    Y = données[:, -1].reshape(-1, 1)

    print("Sorties attendues :")
    print(Y.T)

    test = MLP()
    test.initialisation([3, 5, 3, 1], activation='relu', T=0.01)

    test.train(D, Y, epochs=1000)
    
    
    A, _, _ = test.propagation_avant(D)
    print("Min:", np.min(A))
    print("Max:", np.max(A))
    print("moyenne:", np.mean(A))

    A, activations, Avant_act = test.propagation_avant(D, print_details=True)
    print("\nsortie du réseau : ")
    print(A.T)



if __name__ == "__main__":
    main()