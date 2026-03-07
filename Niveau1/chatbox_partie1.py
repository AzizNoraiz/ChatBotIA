
#Dictionnaire
req_res={"Bonjour":"Comment est ce que je peux d'aider ?","Donne moi toute la liste des métiers lié au domaine de la Data": "Data Scientist, Data Engineer, Data Analyst, ML Engineer, BI Analyst",
"Donne moi la définition d'un data scientist": "Spécialiste des statistiques, de l’informatique et du marketing, le Data Scientist recueille, traite, analyse et fait parler les données massives, autrement appelées “big data” ,dans le but d’améliorer les performances d'une entreprise", 
"Donne moi la définition d'un data analyst" : "Un data analyst, ou analyste de données, est un professionnel spécialisé dans l'extraction, l'interprétation puis l'analyse de grandes quantités de données","Donne moi toute la liste des métiers lié au domaine de la cybersécurité " : "Analyste SOC, Pentester, Responsable de la Sécurité des Systèmes d'Information, Expert en Forensique Numérique",
"Donne moi la définition d'un Analyste SOC" : "Professionnel qui surveille en continu les systèmes informatiques pour détecter et répondre aux incidents de sécurité. Il analyse les alertes, investigate les menaces et coordonne les réponses aux incidents. "
, "Donne moi la définition d'un Pentester" : "Expert qui simule des attaques cybernétiques pour identifier les vulnérabilités des systèmes avant les hackers malveillants. Il réalise des tests d'intrusion et recommande des corrections."
, "Donne moi la définition d'un Responsable de la Sécurité des Systèmes d'Information" : "Cadre qui définit et met en œuvre la stratégie de sécurité globale d'une organisation. Il manage l'équipe sécurité, assure la conformité réglementaire et gère le budget sécurité."
,"Donne moi la définition d'un Expert en Forensique Numérique" : "Spécialiste qui investigate les cyberattaques après leur survenue. Il collecte et analyse les preuves numériques pour identifier les causes et les responsables d'un incident." }

#Fonction qui associe le input rentrer par l'utilisateur à un output
def chatbox(req_res,input):
    if input  in req_res:
        print(req_res[input]+"\n")
    else :
        print("Je ne suis pas en capacité de répondre à cette question !")

#MAIN
print("ChatBot de sensibilation à l'informatique \n")
while True :
    print("Tapez 'quit' pour quitter")
    req1 = input("ChatBot entrer textuel attendue:  ")

    # Attention à bien taper le input !!!
    # Cas 1 si input vaut -> Bonjour
    # Cas 2 si input vaut -> Donne moi toute la liste des métiers lié au domaine de la Data
    # Cas 3 si input vaut -> Donne moi la définition d'un data scientist
    # Cas 4  si input vaut -> Donne moi la définition d'un data analyst
    # Cas 5  si input vaut -> Donne moi toute la liste des métiers lié au domaine de la cybersécurité
    # Cas 6  si input vaut -> Donne moi toute la liste des métiers lié au domaine de la cybersécurité
    # Cas 7 si input vaut -> Donne moi la définition d'un Analyste SOC
    # Cas 8 si input vaut -> Donne moi la définition d'un Pentester
    # Cas 9 si input vaut -> Donne moi la définition d'un Responsable de la Sécurité des Systèmes d'Information
    # Cas 10 si input vaut -> Donne moi la définition d'un Expert en Forensique Numérique

    if req1 == "quit":
        print("Au revoir \n")
        break
    else :
        chatbox(req_res,req1)
