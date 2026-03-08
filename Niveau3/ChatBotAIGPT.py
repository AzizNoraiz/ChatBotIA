# =============================================================
# CHATBOT BIBLIOTHEQUE - NIVEAU 3 : RAG + API OpenAI (GPT)
# =============================================================
# Architecture : RAG = Retrieval-Augmented Generation

#
# Principe :
#   1. On RECUPERE les livres pertinents depuis le JSON (Retrieval)
#      → via TF-IDF (comme Niveau 2)
#   2. On AUGMENTE la requête avec ces livres comme contexte
#      → on les injecte dans le prompt envoyé à GPT
#   3. GPT GENERE une réponse naturelle basée sur ce contexte
#      → il ne "devine" pas, il répond avec NOS données
#
# Installation :
#   
#   python -m spacy download fr_core_news_sm
#
# Clé API :
#   Créer un compte sur https://platform.openai.com
#   Générer une clé API dans Settings > API Keys
# =============================================================

import json
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np

# -------------------------------------------------------------
# CONFIGURATION — Mettre ta clé API ici
# -------------------------------------------------------------
# BONNE PRATIQUE : ne jamais écrire la clé directement dans le code
# On la lit depuis une variable d'environnement du système.
# Dans ton terminal, tape : export OPENAI_API_KEY="sk-..."
# Puis relance le script.
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# -------------------------------------------------------------
# ETAPE 1 — Chargement spaCy + données
# -------------------------------------------------------------
nlp = spacy.load("fr_core_news_sm")

with open("bibliotheque.json", "r", encoding="utf-8") as f:
    bibliotheque = json.load(f)

# -------------------------------------------------------------
# ETAPE 2 — Lemmatisation (identique Niveau 2)
# -------------------------------------------------------------
def lemmatiser(texte):
    doc = nlp(texte.lower())
    lemmes = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct]
    return " ".join(lemmes)

# -------------------------------------------------------------
# ETAPE 3 — Construction de la matrice TF-IDF (identique N2)
# -------------------------------------------------------------
def construire_description(livre):
    return (
        f"{livre['Titre du livre']} "
        f"{livre['Auteur']} "
        f"{livre['Domaine']} "
        f"{livre['Année de publication']} "
        f"{livre['Disponibilité']}"
    )

descriptions = [lemmatiser(construire_description(l)) for l in bibliotheque]
vectoriseur = TfidfVectorizer()
#TfidfVectorizer est une classe de la bibliothèque scikit-learn. Cette ligne crée une instance de cette classe càd un Vecteur 
matrice_tfidf = vectoriseur.fit_transform(descriptions) # Creation de la matrice Livrex

# -------------------------------------------------------------
# ETAPE 4 — Retrieval : récupérer les livres les plus pertinents
# -------------------------------------------------------------
# C'est la partie "R" du RAG.
# On cherche les N livres les plus similaires à la requête
# via la similarité cosinus (même logique que Niveau 2).
# Ces livres seront injectés comme CONTEXTE dans le prompt GPT.

def recuperer_livres_pertinents(requete, top_n=3):
    """
    Retourne les top_n livres les plus pertinents pour la requête.
    top_n=3 : on envoie max 3 livres à GPT pour ne pas surcharger le prompt.
    """
    requete_lemmatisee = lemmatiser(requete)
    vecteur = vectoriseur.transform([requete_lemmatisee])
    similarites = cosine_similarity(vecteur, matrice_tfidf).flatten()
    indices_tries = np.argsort(similarites)[::-1]

    livres_pertinents = []
    for i in indices_tries[:top_n]:  # on prend les top_n meilleurs
        if similarites[i] > 0.0:     # on ignore les score nuls (aucun mot commun)
            livres_pertinents.append(bibliotheque[i])

    return livres_pertinents

# -------------------------------------------------------------
# ETAPE 5 — Construction du Contexte pour GPT
# -------------------------------------------------------------
# C'est la partie "A" (Augmented) du RAG.
# On formate les livres récupérés en texte lisible pour GPT.
# GPT va s'appuyer UNIQUEMENT sur ce contexte pour répondre.
# Il ne piochera pas dans ses connaissances générales pour
# des infos sur les livres — ce qui évite les hallucinations.

def construire_contexte(livres):
    """
    Transforme une liste de livres en texte structuré
    pour l'inclure dans le prompt GPT.
    """
    if not livres:
        return "Aucun livre trouvé dans la bibliothèque."

    contexte = "Voici les livres disponibles dans la bibliothèque :\n\n"
    for i, livre in enumerate(livres, 1):
        contexte += f"Livre {i} :\n"
        contexte += f"  - Titre       : {livre['Titre du livre']}\n"
        contexte += f"  - Auteur      : {livre['Auteur']}\n"
        contexte += f"  - Domaine     : {livre['Domaine']}\n"
        contexte += f"  - Publication : {livre['Année de publication']}\n"
        contexte += f"  - Disponibilité: {livre['Disponibilité']}\n\n"
    return contexte

# -------------------------------------------------------------
# ETAPE 6 — Prompt Système (instructions permanentes pour GPT)
# -------------------------------------------------------------
# Le "system prompt" est le message de configuration envoyé
# à GPT UNE SEULE FOIS au début de la conversation.
# Il définit le ROLE et le COMPORTEMENT du modèle.
# GPT s'y conforme tout au long de la conversation.

SYSTEM_PROMPT = """
Tu es un assistant intelligent pour une bibliothèque universitaire.
Tu aides les étudiants à trouver des livres selon leurs besoins.

Règles strictes que tu dois toujours respecter :
1. Tu réponds UNIQUEMENT en te basant sur le contexte de livres fourni.
2. Si un livre demandé n'est pas dans le contexte, dis-le clairement.
3. Tu réponds toujours en français, de façon naturelle et sympathique.
4. Si un livre est "Emprunté", tu le signales et proposes des alternatives disponibles.
5. Tu peux donner des conseils sur les livres (lequel lire en premier, etc.)
   mais uniquement pour les livres présents dans le contexte.
6. Tu ne dois JAMAIS inventer des titres, auteurs ou informations qui ne sont
   pas dans le contexte fourni.
"""

# -------------------------------------------------------------
# ETAPE 7 — Historique de conversation (mémoire du chatbot)
# -------------------------------------------------------------
# GPT est "sans mémoire" par défaut : chaque appel API est indépendant.
# Pour simuler une vraie conversation, on lui renvoie tout l'historique
# à chaque message. C'est la technique standard pour les chatbots GPT.
#
# Structure : liste de dictionnaires {"role": ..., "content": ...}
# Roles possibles :
#   "system"    → instructions permanentes (notre SYSTEM_PROMPT)
#   "user"      → messages de l'utilisateur
#   "assistant" → réponses de GPT

#historique est une liste de dictionnaire qui a pour clé un role (qui a écrit ce message ) peut etre system,user,assistant
#  et comme valeur associé à la clé le contenue 
historique = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# -------------------------------------------------------------
# ETAPE 8 — Appel à l'API OpenAI (la partie "G" du RAG)
# -------------------------------------------------------------
# C'est ici que GPT génère la réponse.
# On lui envoie :
#   - Le system prompt (rôle + règles)
#   - L'historique complet de la conversation
#   - Le message actuel de l'utilisateur + le contexte de livres

def appeler_gpt(requete_utilisateur, contexte_livres):
    """
    Envoie la requête + contexte à GPT et retourne sa réponse.
    """
    # On combine la requête de l'utilisateur avec le contexte des livres
    # GPT reçoit ainsi les données ET la question en même temps
    message_augmente = f"""
Question de l'étudiant : {requete_utilisateur}

{contexte_livres}

Réponds à la question en te basant uniquement sur les livres ci-dessus.
"""
    # On ajoute le message à l'historique
    historique.append({"role": "user", "content": message_augmente})

    # Appel API OpenAI
    reponse = client.chat.completions.create(
        model="gpt-3.5-turbo",   # modèle léger et économique (≈ 0.001€ par échange)
                                # alternatives : "gpt-4o" (plus puissant, plus cher)
                                #                "gpt-3.5-turbo" (moins cher encore)
        messages=historique,   # historique complet pour la mémoire
        temperature=0.3,       # 0 = réponses très précises et répétables
                               # 1 = réponses créatives et variées
                               # 0.3 = légèrement créatif mais fiable
        max_tokens=500         # limite la longueur de la réponse (économie de tokens)
    )

    # Extraction du texte de la réponse
    texte_reponse = reponse.choices[0].message.content

    # On ajoute la réponse de GPT à l'historique (pour la mémoire)
    historique.append({"role": "assistant", "content": texte_reponse})

    return texte_reponse

# -------------------------------------------------------------
# ETAPE 9 — Fonction principale du chatbot RAG
# -------------------------------------------------------------
def chatbot_rag(requete):
    """
    Pipeline complet RAG :
    Requête → TF-IDF (Retrieval) → Contexte (Augmented) → GPT (Generation)
    """
    print("\n🔍 Recherche des livres pertinents...")

    # R — Retrieval : on récupère les livres les plus pertinents
    livres_pertinents = recuperer_livres_pertinents(requete, top_n=3)

    if livres_pertinents:
        print(f"📚 {len(livres_pertinents)} livre(s) trouvé(s), envoi à GPT...")
    else:
        print("⚠️  Aucun livre pertinent trouvé, GPT va le signaler.")

    # A — Augmented : on formate les livres en contexte textuel
    contexte = construire_contexte(livres_pertinents)

    # G — Generation : GPT génère une réponse naturelle
    print("🤖 GPT génère une réponse...\n")
    reponse = appeler_gpt(requete, contexte)

    return reponse

# -------------------------------------------------------------
# MAIN — Boucle de dialogue
# -------------------------------------------------------------
print("=" * 60)
print("  📚 ChatBot Bibliothèque - Niveau 3 (RAG + GPT)")
print("=" * 60)
print("Exemples de questions naturelles :")
print("  → Quel livre me conseilles-tu pour débuter en IA ?")
print("  → Y a-t-il des livres disponibles sur la programmation ?")
print("  → Qui a écrit Clean Code et est-il disponible ?")
print("  → Je veux apprendre Python, par où commencer ?")
print()

while True:
    print("Tapez \'quitter\' pour quitter")
    req = input("Vous > ").strip()

    if req.lower() == "quitter":
        print("Au revoir !")
        break
    elif req == "":
        continue
    else:
        reponse = chatbot_rag(req)
        print(f"\n🤖 Assistant > {reponse}\n")
        print("-" * 60)
