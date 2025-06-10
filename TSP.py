# -*- coding: utf-8 -*-
"""
Created on Fri May 24 8:04:28 2024

@author: Christophe Duhamel
"""

import random
import math
import itertools
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class Sommet:
    cpt_sommet = 0
    
    def __init__ (self, x, y,couleur):
        self.id = Sommet.cpt_sommet
        Sommet.cpt_sommet += 1
        
        self.x = x
        self.y = y
        # Ajout de la propriété couleur pour un sommet
        self.couleur = couleur
    
    def getId (self):
        return self.id
    
    def getX (self):
        return self.x
    
    def getY (self):
        return self.y
    
    def getCouleur (self):
        return self.couleur
    
    def affiche (self):
        print('({}:{:.2f},{:.2f})'.format(self.id,self.x,self.y))


class Instance:
    def __init__ (self, name, n):
        self.name = name
        self.nb_sommets = n
        self.reset()
    
    def size (self):
        return self.nb_sommets
    
    def reset (self):
        self.generateNodes()
        self.computeDistances()

    
    def generateNodes (self):
        self.sommets = [Sommet(
                        random.uniform(0,100),
                        random.uniform(0,100),
                        # alternance de couleurs pour les sommets
                        "red" if i % 2 == 0 else "black" 
                        ) for i in range(self.nb_sommets)]
    
    def computeDistances (self):
        self.dist = [[0.0] * self.nb_sommets for i in range(self.nb_sommets)]
        for si in self.sommets:
            for sj in self.sommets:
                delta_x = si.getX() - sj.getX()
                delta_y = si.getY() - sj.getY()
                self.dist[si.getId()][sj.getId()] = math.sqrt(delta_x ** 2 + delta_y ** 2)
    
    def affiche (self):
        print('{} sommets: '.format(self.nb_sommets))
        for s in self.sommets:
            print('  ', end='')
            s.affiche()
        print('dist:')
        for line in self.dist:
            for elt in line:
                print(' {:6.2f}'.format(elt), end='')
            print()

    def plot (self):
        plt.figure()
        plt.title('instance {}: {} sommets'.format(self.name, self.nb_sommets))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-1,101)
        plt.ylim(-1,101)
        x = [elt.getX() for elt in self.sommets]
        y = [elt.getY() for elt in self.sommets]
        plt.scatter(x,y)
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def load_instance_from_csv(coord_file, dist_file):
        coords_df = pd.read_csv(coord_file)
        dist_matrix = np.loadtxt(dist_file, delimiter=",")

        sommets = []
        Sommet.cpt_sommet = 0  
        for i, (lat, lon) in enumerate(coords_df.values):
            couleur = "red" if i % 2 == 0 else "black"
            s = Sommet(x=lon, y=lat, couleur=couleur)
            sommets.append(s)

        # Création sans appel à reset()
        instance = Instance.__new__(Instance)  # bypass __init__
        instance.name = "POI"
        instance.nb_sommets = len(sommets)
        instance.sommets = sommets
        instance.dist = dist_matrix
        return instance
    
    def generateColors_from_sequence(self, sequence):
        """
        Attribue les couleurs pour maximiser les conflits sur une séquence donnée.
        Deux sommets consécutifs auront la même couleur.
        """
        for i, idx in enumerate(sequence):
            couleur = "red" if i % 2 == 0 else "red"  # Toujours la même couleur pour créer des conflits
            self.sommets[idx].couleur = couleur

    def generateColors_random(self):
        """
        Attribue aléatoirement 'red' ou 'black' à chaque sommet.
        """
        for sommet in self.sommets:
            sommet.couleur = random.choice(["red", "black"])

    def generateColors_by_position(self):
        """
        Coupe l'ensemble des sommets en deux groupes selon la position X.
        X < 50 : rouge, X >= 50 : noir
        """
        for sommet in self.sommets:
            sommet.couleur = "red" if sommet.x < 50 else "black"



class Solution:
    def __init__ (self, inst, desc):
        self.instance = inst
        self.name = desc
        self.sequence = [-1] * inst.size()
        self.valeur = 0.0
        self.temps = 0.0

    def getSequence (self):
        return self.sequence

    def getValeur (self):
        return self.valeur
    
    def getTemps (self):
        return self.temps
    
    def setTemps (self, t):
        self.temps = t

    def setSequence (self, s):
        self.sequence = s
        self.evalue()
    
    def evalue (self):
        val = 0.0
        for i in range(-1, len(self.sequence)-1):
            val += self.instance.dist[self.sequence[i]][self.sequence[i+1]]
        self.valeur = val
        
    def affiche (self):
        print('solution \'{}\': {} -> val = {:.2f} temps = {:.2f} s'.format(self.name, self.sequence, self.valeur, self.temps))

    def plot(self, savepath=None):
            plt.figure()
            plt.title(f"TSP Rouge/Noir - Distance: {self.valeur:.2f}")
            for i, idx in enumerate(self.sequence):
                sommet = self.instance.sommets[idx]
                color = 'red' if sommet.getCouleur() == "red" else 'black'
                plt.scatter(sommet.getX(), sommet.getY(), c=color, s=100)
                plt.text(sommet.getX(), sommet.getY() + 2, str(idx), ha='center')

            x = [self.instance.sommets[idx].getX() for idx in self.sequence]
            y = [self.instance.sommets[idx].getY() for idx in self.sequence]
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, 'b-', alpha=0.5)
            plt.grid(True)

            if savepath:
                plt.savefig(savepath)
                plt.close()
            else:
                plt.show()



class Heuristiques:
    def __init__ (self, inst):
        self.instance = inst
        self.evolution = []

    def plot(self, savepath=None):
        plt.figure()
        plt.title('Évolution du coût')
        plt.xlabel('temps (s)')
        plt.ylabel('valeur')
        x = [elt[0] for elt in self.evolution]
        y = [elt[1] for elt in self.evolution]
        plt.plot(x, y, marker='o')
        plt.grid(True)

        if savepath:
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()

        
    def compute_triviale(self):
        # Génère une séquence alternée rouge/noir
        seq = []
        red_nodes = [s.getId() for s in self.instance.sommets if s.getCouleur() == "red"]
        black_nodes = [s.getId() for s in self.instance.sommets if s.getCouleur() == "black"]
        
        # Alterne les sommets rouges et noirs
        for r, b in zip(red_nodes, black_nodes):
            seq.extend([r, b])
        
        s = Solution(self.instance, 'triviale (alternée)')
        s.setSequence(seq)
        return s
        
    def compute_random(self):
        red_nodes = [s.getId() for s in self.instance.sommets if s.getCouleur() == "red"]
        black_nodes = [s.getId() for s in self.instance.sommets if s.getCouleur() == "black"]
        
        # Mélange les listes séparément
        random.shuffle(red_nodes)
        random.shuffle(black_nodes)
        
        # Alterne les couleurs en commençant par un rouge ou un noir aléatoire
        seq = []
        start_with_red = random.choice([True, False])
        
        if start_with_red:
            for r, b in zip(red_nodes, black_nodes):
                seq.extend([r, b])
        else:
            for b, r in zip(black_nodes, red_nodes):
                seq.extend([b, r])
        
        s = Solution(self.instance, 'random (alterné)')
        s.setSequence(seq)
        return s
        
    def compute_nearest (self):
        available = [i for i in range(1,self.instance.size())]
        current = 0
        seq = [current]
        # Récuperation de la couleur du sommet de départ
        last_color = self.instance.sommets[current].getCouleur() 
        
        while len(available) != 0:
            best = None
            #dist = 200.0
            # On cherche le plus proche voisin de couleur différente
            dist_min = float('inf')
            for elt in available:
                sommet = self.instance.sommets[elt]
                if sommet.getCouleur() != last_color:  # Contrainte rouge/noir
                    distance = self.instance.dist[current][elt]
                    if distance < dist_min:
                        dist_min = distance
                        best = elt
            if best is None:
                best = available[0]  # Si aucun sommet valide , relâcher la contrainte 
            seq.append(best)
            available.remove(best)
            current = best
            last_color = self.instance.sommets[current].getCouleur()  # Mettre à jour la couleur du dernier sommet
        
        s = Solution(self.instance, 'nearest neighbour')
        s.setSequence(seq)
        return s

    def compute_enumerate (self):
        record = Solution(self.instance, 'enumerate')
        
        if self.instance.size() > 11:
            print('-> too many nodes for the enumeration. Stop.')
            return record

        debut = time.time()
        seq = [i for i in range(self.instance.size())]
        record.setSequence(seq)
        
        s = Solution(self.instance, 'tmp')
        perm = itertools.permutations(seq)
        for p in perm:
            s.setSequence(list(p))
            if s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
        duree = time.time() - debut
        record.setTemps(duree)
        return record


    def mvt2Opt(self, s: Solution):
        seq = s.getSequence().copy() 
        dist = self.instance.dist
        n = len(seq)

        for i in range(1, n - 2):  
            for j in range(i + 1, n - 2):  
                a, b = seq[i - 1], seq[i]
                c, d = seq[j], seq[j + 1]

                # Vérifier alternance rouge/noir après inversion
                if self.instance.sommets[a].getCouleur() == self.instance.sommets[c].getCouleur():
                    continue
                if self.instance.sommets[b].getCouleur() == self.instance.sommets[d].getCouleur():
                    continue

                delta = dist[a][b] + dist[c][d] - dist[a][c] - dist[b][d]
                if delta > 0:
                    # 💡 Modifie une copie et la renvoie
                    new_seq = seq[:i] + list(reversed(seq[i:j+1])) + seq[j+1:]
                    s.setSequence(new_seq)
                    return True
        return False


    
    def localSearch(self, s, max_iter=1000):
        cpt = 0
        while cpt < max_iter and self.mvt2Opt(s):
            cpt += 1
        return cpt

    
    def multistart (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record
    
    def multistart_LS (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart_LS')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            self.localSearch(s,max_iter=500)
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record

def check_alternance(sequence, instance):
    for i in range(len(sequence) - 1):
        current = instance.sommets[sequence[i]].getCouleur()
        next = instance.sommets[sequence[i+1]].getCouleur()
        if current == next:
            return False
    return True

def count_color_conflicts(sequence, instance):
    """Compte le nombre de conflits de couleur dans une séquence."""
    conflits = 0
    for i in range(len(sequence) - 1):
        c1 = instance.sommets[sequence[i]].getCouleur()
        c2 = instance.sommets[sequence[i+1]].getCouleur()
        if c1 == c2:
            conflits += 1
    return conflits

if __name__ == '__main__':
    random.seed(42)  # Pour la reproductibilité des résultats

    # ======================================
    # TEST SUR INSTANCE ALÉATOIRE (20 sommets)
    # ======================================
    inst = Instance('RougeNoir', 20)  # 20 sommets avec couleurs alternées
    inst.plot()
    heur = Heuristiques(inst)

     # 1. Générer une solution TSP classique (par exemple, nearest neighbour)
    s_tsp = heur.compute_nearest()
    sequence_tsp = s_tsp.getSequence()

    # 2. Attribuer les couleurs pour maximiser les conflits sur cette séquence
    inst.generateColors_from_sequence(sequence_tsp)
    print("Conflits de couleurs générés sur la séquence TSP.")

    # 3. Compter le nombre de conflits de couleur dans la solution TSP classique
    nb_conflits = count_color_conflicts(sequence_tsp, inst)
    print(f"Nombre de conflits de couleur dans la solution TSP classique : {nb_conflits}")

    # 4. (Optionnel) Générer des couleurs aléatoires
    inst.generateColors_random()
    print("Couleurs aléatoires générées.")

    # 5. (Optionnel) Générer des couleurs par position
    inst.generateColors_by_position()
    print("Couleurs générées par position X.")

    # Heuristique plus proche voisin
    # debut = time.time()
    # s3 = heur.compute_nearest()
    # duree = time.time() - debut
    # print('heuristique plus proche voisins: duree = {:.3f} s'.format(duree))
    # s3.affiche()
    # s3.plot()

    # Recherche locale
    # debut = time.time()
    # heur.localSearch(s3)
    # duree = time.time() - debut
    # print('recherche locale: duree = {:.3f} s'.format(duree))
    # s3.affiche()
    # s3.plot()

    # Test de toutes les méthodes (décommenter si besoin)
    # methodes = [heur.compute_triviale, heur.compute_random, heur.compute_nearest, heur.multistart, heur.multistart_LS]
    # for m in methodes:
    #     debut = time.time()
    #     sol = m()
    #     duree = time.time() - debut
    #     sol.setTemps(duree)
    #     sol.affiche()
    #     check = check_alternance(sol.getSequence(), inst)
    #     print(f"Alternance respectée: {check}")  # Doit retourner True si l'alternance est respectée
    #     sol.plot()
    #     print('evolution = ', heur.evolution)
    #     if len(heur.evolution) > 0:
    #         heur.plot()

    # print("\n" + "="*50)
    # print("Test sur instances réelles à partir des POIs OSM")
    # print("="*50)

    # ======================================
    # CHARGEMENT D'UNE INSTANCE RÉELLE
    # ======================================
    # instance = Instance.load_instance_from_csv(
    #     coord_file='pois_coords.csv',
    #     dist_file='dist_matrix.csv'
    # )
    # heur = Heuristiques(instance)
    # methodes = [heur.compute_triviale, heur.compute_random, heur.compute_nearest, heur.multistart, heur.multistart_LS]
    # heur = Heuristiques(instance)
    # for m in methodes:
    #     debut = time.time()
    #     sol = m()
    #     duree = time.time() - debut
    #     sol.setTemps(duree)
    #     sol.affiche()
    #     sol.plot()
    #     print(check_alternance(sol.getSequence(), instance))  # Doit retourner True si l'alternance est respectée

    #     sol.plot(savepath=f"figures/solution_{m.__name__}.png")

    #     if len(heur.evolution) > 0:
    #         heur.plot(savepath=f"figures/evolution_{m.__name__}.png") 

