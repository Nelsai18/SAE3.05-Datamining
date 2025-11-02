import os
print(os.getcwd())

import folium
import os

# Exemple : enregistrer sur le Bureau (Windows ou macOS)
path = os.path.expanduser("C:/Users/Fortunato/Documents/BUT3/SAE_503_data_mining/titanic_route.html")

# Coordonnées (lat, lon)
southampton = (50.9097, -1.4044)
cherbourg   = (49.6300, -1.6200)
cobh        = (51.8503, -8.2943)       # Queenstown / Cobh
wreck       = (41.7256, -49.9471)      # Lieu du naufrage (approx.)
new_york    = (40.7128, -74.0060)

# Liste d'étapes dans l'ordre du trajet (approximatif)
route_coords = [southampton, cherbourg, cobh, wreck, new_york]

# Centre de la carte (centrer sur l'Atlantique nord)
m = folium.Map(location=[48.0, -30.0], zoom_start=3, tiles='CartoDB positron')

# Ajouter marqueurs avec popup
folium.Marker(southampton, tooltip='Southampton (départ)').add_to(m)
folium.Marker(cherbourg,   tooltip='Cherbourg (escale)').add_to(m)
folium.Marker(cobh,        tooltip='Queenstown / Cobh (escale)').add_to(m)
folium.Marker(wreck,       tooltip='Lieu du naufrage (approx.)',
              popup='Lieu du naufrage (~41.7256, -49.9471)').add_to(m)
folium.Marker(new_york,    tooltip='New York (destination visée)').add_to(m)

# Tracer la ligne de route (polyline)
folium.PolyLine(route_coords, color='blue', weight=3.5, opacity=0.7).add_to(m)

# Optionnel : ajouter des cercles pour le wreck (pour bien le repérer)
folium.Circle(location=wreck, radius=30000, color='red', fill=True, fill_opacity=0.1,
              popup='Zone approximative du naufrage').add_to(m)

# Sauvegarder en HTML
m.save('titanic_route.html')
print("Carte sauvegardée dans 'titanic_route.html'. Ouvre ce fichier dans ton navigateur.")
