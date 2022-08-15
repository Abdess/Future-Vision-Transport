pays_propose = input("Choisis un pays parmis les suivants et je te donne sa\
capitale : France, Espagne, Portugal, Bresil et Suede : ")

liste_pays = ["France", "Espagne", "Portugal", "Bresil", "Suede"]
liste_capitales = ["Paris", "Madrid", "Lisbonne", "Brasilia", "Stockholm"]


def nom_capitale(nom_pays):
    for pays in liste_pays:
        if nom_pays == pays:
            return liste_capitales[liste_pays.index(pays)]
        else:
            return f"La capitale de {nom_pays} est introuvable :'("


print(nom_capitale(pays_propose))
