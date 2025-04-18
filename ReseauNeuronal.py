import numpy as np


fichier_test = "données.txt"  
données = np.loadtxt(fichier_test)

D = données[:, :-1]  
dernière_colonne = données[:, -1]    



class MLP:

    def initialisation(self, nbr_neurones , activation ='relu', T=0.01):

        print("liste du nombre de neurones par couche:")
        self.nbr_neurones = nbr_neurones

        print("la fonction d'activation est : ")
        self.activation = self.relu

        print("la dérivée de la fonction d'activation est :")
        self.activation_dérivée =self.relu_dérivée
        
        print("le taux d'apprentissage est ")
        self.T = T
        
        
        print("la liste de matrices de poids aléatoires")
        self.poids = [np.random.randn(nbr_neurones[i + 1],nbr_neurones[i])* np.sqrt(2 / nbr_neurones[i]) for i in range(len(nbr_neurones) - 1)]
        
        print("la liste de vecteurs biais initialisés à 0 :")
        self.biases = [np.zeros((nbr_neurones[i + 1], 1)) for i in range(len(nbr_neurones) - 1)]


    