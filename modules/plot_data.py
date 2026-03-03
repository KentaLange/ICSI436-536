import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def plot_corr(df, feature_names, dataset_name):
    corr = df[feature_names].corr().to_numpy()
    plt.figure()
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"{dataset_name} — Feature Correlation")
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.yticks(range(len(feature_names)), feature_names)

    # Add correlation numbers in each cell (compact but informative)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(label="Correlation")
    plt.tight_layout()
    plt.savefig(rf"c{dataset_name}_corr.png")
    plt.show()



