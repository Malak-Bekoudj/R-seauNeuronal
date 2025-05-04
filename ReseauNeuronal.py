import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


class MLP:

    def initialisation(self, nbr_neurones , activation ='relu', T=0.01):

        self.nbr_neurones = nbr_neurones              
        self.activation = self.relu
        self.activation_dérivée =self.relu_dérivée           
        self.T = T
        self.poids = [np.random.randn(nbr_neurones[i + 1], nbr_neurones[i]) * np.sqrt(2 / nbr_neurones[i]) for i in range(len(nbr_neurones) - 1)]
        self.biases = [np.zeros((nbr_neurones[i + 1], 1)) for i in range(len(nbr_neurones) - 1)]


    def relu(self, Z):
      return np.maximum(0, Z)
    
    def relu_dérivée(self, Z):
      return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_dérivée(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def propagation_avant(self,X):
       A = X.T
       activations = [A]
       Avant_act = []

       for i in range(len(self.poids)):
            Z = np.dot(self.poids[i], A) + self.biases[i]
            Avant_act.append(Z)
            A = self.sigmoid(Z) if i == len(self.poids) - 1 else self.activation(Z)
            activations.append(A)
       return A, activations, Avant_act
    
    def rétropropagation(self, X, Y):
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

    def train(self, X, Y, epochs=2000):
            
      for epoch in range(epochs):
        A, _, _ = self.propagation_avant(X)
        loss = np.mean((A.T - Y)**2)
        self.rétropropagation(X, Y)
        if epoch % 100 == 0:
           print(f"Epoch {epoch}: Loss = {loss:.5f}")


    def prédire(self, X):      
        A, _, _ = self.propagation_avant(X)
        return A.T
    
    def précision(self, X, Y):
       
        prédictions = np.round(self.prédire(X))
        exactes = (prédictions == Y).sum()
        total = Y.shape[0]
        pourcentage = (exactes / total) * 100
        return pourcentage
    
    def load_predict(self, test_file_path):
       
        test_data = np.loadtxt(test_file_path)        
        if test_data.ndim == 1:
            test_data = test_data.reshape(1, -1)

        if test_data.shape[1] == self.nbr_neurones[0] + 1:
            test_data = test_data[:, :-1]
                         
        return self.prédire(test_data)
       
      
def main():     
 
    données = np.loadtxt("data.txt")
    D = données[:, :-1]
    Y = données[:, -1].reshape(-1, 1)

    """
    print("\n XOR :")
    E = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    S = np.array([[0], [1], [1], [0]])
    xor_mlp = MLP()
    xor_mlp.initialisation([2, 5, 1], activation='relu', T=0.1)
    xor_mlp.train(E, S, epochs=5000)
    sortie_XOR = xor_mlp.prédire(E)
    print("Sorties du MLP (XOR):\n", np.round(sortie_XOR))
    """   
    mlp = MLP()
    mlp.initialisation([D.shape[1], 10, 5, 1], activation='relu', T=0.005)
    mlp.train(D, Y, epochs=20000)
          
    précision = mlp.précision(D, Y)
    print(f"\n Pourcentage d’apprentissage : {précision:.2f} %")
    predictions = mlp.load_predict("test.txt")
    
    if predictions.size > 0:
        np.savetxt("test_bekoudj.txt", predictions)
   
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    X_vase = D[Y.flatten() == 1]
    X_points = D[Y.flatten() == 0]

    #ax.scatter(X_points[:, 0], X_points[:, 1], X_points[:, 2], c='blue', label='Classe 0')
    ax.scatter(X_vase[:, 0], X_vase[:, 1], X_vase[:, 2], c='red', label='Vase (classe 1)')
        
    ax.set_title("Visualisation 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()