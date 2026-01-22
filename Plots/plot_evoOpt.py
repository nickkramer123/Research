# PLOTS for evoOpt Output that can be done after running
# TODO FIX
# best fitness plotted against b
import matplotlib.pyplot as plt


# import best_scores
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



# Create b values
b_values = [i * 0.002 for i in range(len(best_scores))]

# Plot with chaos as color
fig, ax = plt.subplots()
scatter = ax.scatter(b_values, best_scores, c=best_chaos, 
                     cmap='viridis', alpha=0.6, s=50)

ax.set_xlabel('b value')
ax.set_ylabel('best fitness')

# Add colorbar to show chaos scale
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Chaos Measure (Lyapunov)')

plt.savefig("ChoasVsSynergy1.png")
plt.show()