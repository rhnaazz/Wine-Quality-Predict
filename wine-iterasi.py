import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk

# Load dan preprocessing data
data = pd.read_csv("winequality.csv", sep=';')
X = data.drop('quality', axis=1)
y = data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implementasi K-means manual
class ManualKMeans:
    def __init__(self, n_clusters=3, max_iter=4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroid_history = []
        self.label_history = []
        self.accuracies = []
        
    def fit(self, X):
        # Inisialisasi centroid
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.centroid_history.append(self.centroids.copy())
        
        for iter in range(self.max_iter):
            
            print(f'iterasi ke {iter+1}')
            
            # Assign cluster
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            self.label_history.append(labels)
            
            # Update centroid
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            self.centroid_history.append(new_centroids.copy())
            self.centroids = new_centroids
            
            # Hitung akurasi
            self._calculate_accuracy(X, labels, y)
    
    def _calculate_accuracy(self, X, labels, y):
        # Tambahkan label cluster sebagai fitur baru
        X_with_clusters = np.hstack((X, labels.reshape(-1,1)))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_clusters, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Training dan evaluasi model
        model = MLPClassifier(hidden_layer_sizes=(256,256), max_iter=500)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        self.accuracies.append(acc)

# Inisialisasi dan training

iterasi = 1
cluster = 2

kmeans = ManualKMeans(n_clusters=cluster, max_iter=iterasi)
kmeans.fit(X_scaled)

# Reduksi dimensi untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fungsi untuk membuat plot dalam window Tkinter
def create_plot(iterasi):
    fig = plt.figure(figsize=(20, 4 * iterasi))  # Ukuran figure menyesuaikan jumlah iterasi
    gs = GridSpec(nrows=iterasi, ncols=4, figure=fig, height_ratios=[1]*iterasi, width_ratios=[2,1,1,0.5])

    for i in range(iterasi):
        # Kolom 1: Text iterasi
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.axis('off')
        text_content = f"Iterasi {i+1}\nCentroid:\n{np.round(kmeans.centroid_history[i], 2)}"
        ax_text.text(0.1, 0.5, text_content, fontsize=10, va='center')
        
        # Kolom 2: Pergerakan centroid
        ax_cent = fig.add_subplot(gs[i, 1])
        for k in range(kmeans.n_clusters):
            # Ambil centroid history dan transformasikan ke PCA
            cent_path = np.array([pca.transform(cent[k].reshape(1, -1)) for cent in kmeans.centroid_history[:i+1]])
            ax_cent.plot(cent_path[:, 0, 0], cent_path[:, 0, 1], marker='o', label=f'Centroid {k+1}')
        ax_cent.set_title(f'Pergerakan Centroid Iterasi {i+1}', fontsize=10)
        ax_cent.legend()
        
        # Kolom 3: Visualisasi cluster
        ax_clust = fig.add_subplot(gs[i, 2])
        labels = kmeans.label_history[i]
        for k in range(kmeans.n_clusters):
            cluster_data = X_pca[labels == k]
            ax_clust.scatter(cluster_data[:,0], cluster_data[:,1], alpha=0.5, label=f'Cluster {k+1}', s=10)
        ax_clust.scatter(pca.transform(kmeans.centroid_history[i])[:,0], 
                        pca.transform(kmeans.centroid_history[i])[:,1], 
                        marker='x', s=80, c='black', label='Centroid')
        ax_clust.set_title(f'Sebaran Data Iterasi {i+1}', fontsize=10)
        
        # Kolom 4: Akurasi
        ax_acc = fig.add_subplot(gs[i, 3])
        ax_acc.axis('off')
        acc_text = f"Akurasi : {kmeans.accuracies[i]:.4f}" if i < len(kmeans.accuracies) else "N/A"
        ax_acc.text(0.5, 0.5, acc_text, fontsize=10, ha='center', va='center')

    plt.tight_layout()
    return fig

# Fungsi untuk membuat window scrollable
def create_scrollable_window(iterasi):
    root = tk.Tk()
    root.title("K-means Clustering Visualization")

    # Mengatur ukuran window agar tidak melebihi layar
    window_width = 2000  # Lebar window
    window_height = 800  # Tinggi window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Membuat canvas dan scrollbar
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    # Konfigurasi scroll
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Menambahkan plot ke scrollable frame
    fig = create_plot(iterasi)
    canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Menambahkan toolbar
    toolbar = NavigationToolbar2Tk(canvas_fig, scrollable_frame)
    toolbar.update()

    # Menempatkan canvas dan scrollbar di window
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Menjalankan aplikasi
    root.mainloop()

# Membuat window scrollable dengan jumlah iterasi
create_scrollable_window(iterasi=iterasi)
