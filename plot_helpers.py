from modules import *

def plot_counts_ordered(counts):
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    keys = [k for k, v in items]
    vals = [v for k, v in items]

    plt.figure(figsize=(18, 6))  
    plt.bar(range(len(vals)), vals)

    plt.xticks(range(len(keys)), keys, rotation='vertical', fontsize=6)

    plt.tight_layout()
    plt.show()

def nice_plotter(data_x = [], data_y = [], label_x = "", label_y = "", label_title = "", save = False, label_save_title = ""):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'lines.markersize': 8,  # Larger marker size
        'lines.linewidth': 1.5
    })

    plt.figure(figsize=(10, 6), dpi=100) 

    plt.plot(data_x, data_y, "-x",
            markersize=8,
            markeredgewidth=1.5,
            color='blue')

    plt.xlabel(label_x, fontsize=14, fontname='Times New Roman')
    plt.ylabel(label_y, fontsize=14, fontname='Times New Roman')
    plt.title(label_title, fontsize=16, fontname='Times New Roman')

    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if save:
        plt.savefig(label_save_title + ".pdf", bbox_inches='tight')

    plt.show()