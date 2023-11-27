import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

# Carregar dados
df = pd.read_csv("/kaggle/input/coffee-quality-data-cqi/df_arabica_clean.csv")

# Selecionar colunas desejadas
df = df[["Country of Origin", "Altitude", "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
         "Uniformity", "Clean Cup", "Sweetness"]]

# Calcular médias e desvios padrão
med = round(df.iloc[:, 2:8].mean(), 3)
sd = round(df.iloc[:, 2:8].std(), 3)
med_df = pd.DataFrame({"Average": med, "Dev": sd})

# Gráfico de pares
sns.pairplot(df.iloc[:, 2:8])
plt.show()

# Matriz de correlação
cor_matrix = df.iloc[:, 2:8].corr()

# Gráfico de correlação
sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", square=True)
plt.show()

# Análise de Componentes Principais (PCA)
datos_pca = PCA(n_components=2)
pca_resultados = datos_pca.fit_transform(df.iloc[:, 2:8])

# Porcentagem de variação explicada
propvar = datos_pca.explained_variance_ratio_ * 100
cumvar = np.cumsum(propvar)

# Matriz de lambdas
matlambdas = pd.DataFrame({"Eigenvalues": datos_pca.singular_values_ ** 2,
                            "Proportion of Variance": propvar,
                            "Cumulative Variance": cumvar})

# Pontuações PCA
pca_scores = pd.DataFrame(pca_resultados, columns=["PC1", "PC2"])

# Gráfico de dispersão PCA
plt.scatter(pca_scores["PC1"], pca_scores["PC2"], c=np.sum(df.iloc[:, 2:8], axis=1), cmap="coolwarm", s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Sum")
plt.show()

# Hierarquia de agrupamento
destand = (df.iloc[:, 2:8] - df.iloc[:, 2:8].mean()) / df.iloc[:, 2:8].std()
res = linkage(destand.transpose(), method="ward")

# Dendrograma
plt.figure(figsize=(10, 5))
dendrogram(res, orientation="top", distance_sort="descending", show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Euclidean Distance")
plt.show()

# Determinar grupos
grupo = cut_tree(res, n_clusters=2).flatten()

# Gráfico de dispersão com grupos
plt.scatter(pca_scores["PC1"], pca_scores["PC2"], c=grupo, cmap="viridis", s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA with Clusters")
plt.colorbar(label="Group")
plt.show()

# Calcular médias por grupo
df["Group"] = grupo
g1 = df[df["Group"] == 0].iloc[:, 2:8]
g2 = df[df["Group"] == 1].iloc[:, 2:8]

print("Group 1 Mean:")
print(g1.mean())
print("\nGroup 2 Mean:")
print(g2.mean())