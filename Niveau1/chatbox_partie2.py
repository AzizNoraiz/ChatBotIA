import json
import difflib
import unicodedata

# Chargement du fichier JSON
with open("bibliotheque.json", "r") as f:
    bibliotheque = json.load(f)

# Normalisation : minuscules + suppression des accents
def normaliser(texte):
    texte = texte.lower()
    texte = unicodedata.normalize("NFD", texte)#Le problème de fond : comment Unicode stocke les accents
    #NFD signifie Normalization Form Decomposed — décomposition canonique. Cette fonction éclate chaque caractère 
    #accentué en deux parties séparées : "é" <=> e + accent
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return texte

# Recherche floue sur une valeur
def correspondance_floue(valeur, texte_utilisateur, seuil=0.6):
    valeur_norm = normaliser(valeur)
    texte_norm = normaliser(texte_utilisateur)
    # Recherche partielle (contenu dans)
    if valeur_norm in texte_norm or texte_norm in valeur_norm:
        return True
    # Recherche floue (tolérance aux fautes de frappe)
    score = difflib.SequenceMatcher(None, valeur_norm, texte_norm).ratio()
    return score >= seuil

# Détection de l'intention et recherche dans la bibliothèque
def chatbot(bibliotheque, user_input):
    input_norm = normaliser(user_input)
    resultats = []

    for livre in bibliotheque:
        # Intention : recherche par auteur
        if any(mot in input_norm for mot in ["auteur", "ecrit", "ecrit par", "de"]):
            if correspondance_floue(livre["Auteur"], user_input):
                resultats.append(livre)

        # Intention : recherche par titre
        elif any(mot in input_norm for mot in ["titre", "livre", "ouvrage", "appele", "nomme"]):
            if correspondance_floue(livre["Titre du livre"], user_input):
                resultats.append(livre)

        # Intention : recherche par domaine
        elif any(mot in input_norm for mot in ["domaine", "sujet", "categorie", "matiere"]):
            if correspondance_floue(livre["Domaine"], user_input):
                resultats.append(livre)

        # Intention : recherche par année
        elif any(mot in input_norm for mot in ["annee", "publication", "date", "sorti"]):
            if correspondance_floue(str(livre["Année de publication"]), user_input):
