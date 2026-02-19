import matplotlib.pyplot as plt
import csv
import os

nom = input("Titre: ")

episode = []
temps_episodes = []
temps_episodes_moyen = []
record = []

nom = os.path.join("./resultats", f"{nom}.csv")

with open(nom, "r") as f:
    file = csv.reader(f)
    titres = next(file)
    for ligne in file:
        episode.append(float(ligne[0]))
        temps_episodes.append(float(ligne[1]))
        temps_episodes_moyen.append(float(ligne[2]))
        record.append(float(ligne[3]))

plt.close()
plt.xlabel("Episode")
plt.plot(episode, temps_episodes, color="r", label=titres[1])
plt.plot(episode, temps_episodes_moyen, color="g", label=titres[2])
plt.plot(episode, record, color="y", label=titres[3])
plt.legend()
plt.show()