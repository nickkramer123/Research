import matplotlib.pyplot as plt

best_scores = []
with open('best_scores.txt', 'r') as file:
    for line in file:
        processed_line = line.strip()
        best_scores.append(float(processed_line))


# import best_chaos
best_chaos = []
with open('chaos_scores.txt', 'r') as file:
    for line in file:
        processed_line = line.strip()
        best_chaos.append(float(processed_line))



plt.scatter(best_scores, best_chaos)
plt.show()
