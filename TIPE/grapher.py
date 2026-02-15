import matplotlib.pyplot as plt
import csv

nom = input("Titre: ")

episode = []
temps_episodes = []
temps_episodes_moyen = []
record = []

with open(f"{nom}.csv", "r") as f:
    file = csv.reader(f)

    titres = next(file)
    for ligne in file:
        episode.append(ligne[0])
        temps_episodes.append(ligne[1])
        temps_episodes_moyen.append(ligne[2])
        record.append(ligne[3])

plt.close()
plt.xlabel("Episode")
plt.plot(episode, temps_episodes, color="r", label=titres[1])
plt.plot(episode, temps_episodes_moyen, color="g", label=titres[2])
plt.plot(episode, record, color="y", label=titres[3])
plt.legend()
plt.show()