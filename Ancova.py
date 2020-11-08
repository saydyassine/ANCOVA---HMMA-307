#!/usr/bin/env python
# coding: utf-8

# # HMMA 307 : Analyse de la covariance sur les données Crickets de Walker 
# ## Travail fait par : SAYD Yassine 
# ### M2 MIND 2020-2021

# ## 1. Importation et nettoyage des données :

# In[2]:


import numpy as np
import pandas as pd
import csv


# In[3]:


data = pd.read_csv("C:/Users/33665/Desktop/Crickets.csv")
#Importation des données


# In[4]:


print(type(data))
#Format des données


# In[5]:


print(data.shape)
#Taille du dataframe


# In[6]:


print(data)
#Affichage des données (il y a une colonne en trop)


# In[7]:


Crickets = data.drop(["Unnamed: 0"] , axis = 1)
#Supression de la 1ère colonne (inutile)


# In[8]:


print(Crickets)
#Affichage du dataframe


# ## 2. Analyse descriptive :

# In[9]:


print(Crickets.dtypes)
#Nature des variables
#TempEx correspond à la Températue en degrès Celsus chez les grillons Oecanthus exclamationis 
#TempEx correspond à la Températue en degrès Celsus chez les grillons Oecanthus niveus
#ImpulsionEx correspond au nombre d'impulsions de chant par seconde chez les grillons Oecanthus exclamationis
#ImpulsionNiv correspond au nombre d'impulsions de chant par seconde chez les grillons Oecanthus niveus


# In[10]:


print(Crickets.describe(include='all'))
#Summary des varaibles (nombre total,moyenne,écart-type,min,1er quartile,médiane,3ème quartile,max)


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


Crickets.hist(column='TempEx', color='red')
Crickets.hist(column='TempNiv', color='blue')
Crickets.hist(column='ImpulsionEx', color='yellow')
Crickets.hist(column='ImpulsionNiv', color='green')
#Histogrammes des 4 variables


# In[13]:


Crickets['TempEx'].plot.kde()
Crickets['TempNiv'].plot.kde()
#Comparaison de la distribution de la température chez les Oecanthus exclamationis (bleu) et chez les Oecanthus niveus (orange)


# In[14]:


Crickets['ImpulsionEx'].plot.kde()
Crickets['ImpulsionNiv'].plot.kde()
#Comparaison de la distribution du nombre d'impulsions de chant chez les Oecanthus exclamationis (bleu) et chez les Oecanthus niveus (orange)


# In[15]:


Crickets.plot.scatter(x='ImpulsionEx',y='TempEx', color='red')
Crickets.plot.scatter(x='ImpulsionNiv',y='TempNiv', color='blue')
#Tracé des scattergraphs du nombre d'impulsions en fonction de la température


# ## 3. Test de Student :

# In[46]:


import scipy.stats as stats


# Dans un premier temps, on ignore les températures et on compare simplement les fréquences d’impulsion moyennes chez les 2 groupes.
# A cause des valeurs manquantes (NaN), nous allons taper les valeurs à la main.

# In[43]:


ImpEx= [67.9,65.1,77.3,78.7,79.4,80.4,85.8,86.6,87.5,89.1,98.6,100.8,99.3,101.7]
ImpNiv= [44.3,47.2,47.6,49.6,50.3,51.8,60.0,58.5,58.9,60.7,69.8,70.9,76.2,76.1,77.0,77.7,84.7]


# In[44]:


#On effectue le test de Student pour vérifier l'égalité des moyennes des fréquences d’impulsion chez les 2 groupes
scipy.stats.ttest_ind(ImpEx,ImpNiv)


# La p-value de l'ordre de 2×10−5, donc on rejette l'hypothèse nulle. Oecanthus exclamationis a un taux plus élevé que Oecanthus niveus, et la différence des moyennes est très significatives (23.2).

# In[22]:


#On peut également utiliser cette commande-là pour calculer les moyennes chez les 2 groupes : 
print(Crickets['ImpulsionEx'].mean())
print(Crickets['ImpulsionNiv'].mean())


# ## 4. Ancova : 

# Comme la température moyenne pour les mesures Oecanthus exclamationis était de 3,6 °C plus élevé que pour Oecanthus niveus, il est insensé de négliger la température. Oecanthus exclamationis pourrait avoir un taux plus élevé que Oecanthus niveus à certaines températures, mais pas d’autres. 

# On peut contrôler la température avec l’ancova, qui nous dira si la ligne de régression pour Oecanthus exclamationis est plus élevée que la ligne pour Oecanthus niveus; si c’est le cas, cela signifie que Oecanthus exclamationis aurait un taux de pouls plus élevé à n’importe quelle température.

# In[204]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels
import seaborn as sns
sns.set_style("darkgrid")


# In[102]:


#Régression OLS chez les Oecanthus exclamationis
model1 = ols("ImpulsionEx~TempEx",data=Crickets).fit()
print(model1.summary())


# Y_1 = 3.75X_1 - 11.04 où Y_1: Le nombre d'impulsions par seconde chez les Oecanthus exclamationis et X-1: Température en degrès Celsus

# In[108]:


aov1 = sm.stats.anova_lm(model1, type=2)
print(aov1)


# p-value = 1.100375e-10 

# In[103]:


#Tracé des lignes de régression chez les Oecanthus exclamationis 
ax1 = sns.regplot(x="TempEx", y="ImpulsionEx", data=Crickets, color='r')


# In[87]:


#Régression OLS chez les Oecanthus niveus
model2 = ols("ImpulsionNiv~TempNiv",data=Crickets).fit()
print(model2.summary())


# Y_2 = 3.52X_1 - 15.40 où Y_2: Le nombre d'impulsions par seconde chez les Oecanthus niveus et X_2: Température en degrès Celsus

# In[93]:


aov2 = sm.stats.anova_lm(model2)
print(aov2)


# p-value = 1.565616e-15

# In[104]:


#Tracé des lignes de régression chez les Oecanthus niveus
ax2 = sns.regplot(x="TempNiv", y="ImpulsionNiv", data=Crickets, color='b')


# In[105]:


#Superposition des lignes de régression chez les Oecanthus exclamationis (rouge) et chez les Oecanthus niveus (bleu)
ax1 = sns.regplot(x="TempEx", y="ImpulsionEx", data=Crickets, color='r')
ax2 = sns.regplot(x="TempNiv", y="ImpulsionNiv", data=Crickets, color='b')


# La ligne de régression pour Oecanthus exclamationis est plus élevée que la ligne pour Oecanthus niveus; cela signifie que Oecanthus exclamationis aurait un taux de pouls plus élevé à n’importe quelle température.

# La première hypothèse nulle de l’ancova est que les pentes des lignes de régression sont toutes égales; en d’autres termes, que les lignes de régression sont parallèles les unes aux autres. On va acceptez l’hypothèse nulle selon laquelle les lignes de régression sont parallèles et nous testerons la deuxième hypothèse nulle : que les interceptions des lignes de régression sont toutes les mêmes.

# Les pentes ne sont pas significativement différentes (P=0,25); la pente commune est de 3,60, ce qui se trouve entre les pentes pour les lignes séparées (3,52 et 3,75). Sur cette partie-là, je n'ai pas réussi à tester cette hypothèse.

# Ancova fait les mêmes hypothèses que la régression linéaire : normalité et homoscédasticité de Y pour chaque valeur de X, et indépendance. Vérifions au moins l'hypothèse de normalité. 

# In[202]:


#Test de Shapiro chez les Oecanthus exclamationis
stats.shapiro(model1.resid)


# W= 0.9727, p= 0.9105 donc les résidus sont bien distribués suivant la loi normale chez les Oecanthus exclamationis

# In[203]:


#Test de Shapiro chez les Oecanthus niveus
stats.shapiro(model2.resid)


# W= 0.9159, p= 0.1259 donc les résidus sont bien distribués suivant la loi normale chez les Oecanthus niveus

# Maintenant procédons à un test de Tukey sous l'hypothèse que leurs pentes sont toutes les mêmes

# In[188]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[189]:


cm1 = pairwise_tukeyhsd(Crickets['ImpulsionEx'], Crickets['TempEx'], alpha = 0.05)
print (cm1)


# Le test de Tukey ne rejette pas l'hypothèse nulle chez les Oecanthus exclamationis. Donc pas de différences significatives au niveau des intercepts entre chaque paire de lignes

# In[174]:


cm2 = pairwise_tukeyhsd(Crickets['ImpulsionNiv'], Crickets['TempNiv'], alpha = 0.05)
print(cm2)


# Le test de Tukey rejette quasiment à chaque fois l'hypothèse nulle chez les Oecanthus niveus. Donc il y a des différences significatives au niveau des intercepts entre la plupart des paires de lignes
