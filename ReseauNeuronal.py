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