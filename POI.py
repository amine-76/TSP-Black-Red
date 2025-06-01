import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np

# -----------------------------
# 1. Configuration de base
# -----------------------------

ox.settings.log_console = True
ox.settings.use_cache = True

# -----------------------------
# 2. Chargement de la ville et du graphe
# -----------------------------
place = "Le Havre, France"
print(f"Chargement de la ville : {place}")
graph = ox.graph_from_place(place, network_type='drive')

# -----------------------------
# 3. Chargement des points d'intérêt (POI)
# -----------------------------
print("Chargement des points d'intérêt (POI)...")
tags = {"amenity": "cafe"} # modifiez les tags pour d'autres POI
gdf_pois = ox.features_from_place(place, tags=tags)
                                  
# Filtrer pour avoir des POIs avec coordonnées valides
gdf_pois = gdf_pois[["geometry"]].dropna()

# -----------------------------
# 4. Limiter à 20 POIs et les projeter sur le graphe
# -----------------------------
gdf_pois = gdf_pois.head(20)
pois_coords = [(point.y, point.x) for point in gdf_pois.geometry]

# Trouver les nœuds les plus proches dans le graphe
print("Recherche des nœuds les plus proches pour les POIs...")
poi_nodes = [ ox.distance.nearest_nodes(graph, lon, lat) for lat, lon in pois_coords]

#-------------------------------
# 5. Matrice de distance entre les POIs
# -------------------------------
print("Calcul de la matrice de distance entre les POIs...")
lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='length'))
n = len(poi_nodes)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
       try: 
          dist_matrix[i][j] = lengths[poi_nodes[i]][poi_nodes[j]]
       except KeyError:
          dist_matrix[i, j] = float('inf') # Si pas de chemin, distance infinie

# -----------------------------
# 6. Sauvegarde des données
# -----------------------------
print("Sauvegarde des données...")
coords_df = pd.DataFrame(pois_coords, columns=['lat', 'long'])
coords_df.to_csv('pois_coords.csv', index=False)

# Matrice de distances
np.savetxt('dist_matrix.csv', dist_matrix, delimiter=',')

print("✅ Terminé : 20 POIs traités et données sauvegardées.")