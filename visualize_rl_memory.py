"""
128D Episodic Memory Retrieval Visualization.
Generates an English dashboard showing exactly how the k-NN agent corrects the CNN
by retrieving the 15 nearest memories (images) in the 128D latent space.
Also generates individual images for each zone for easier explanation.
"""

import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import NearestNeighbors

from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

LIMIAR_MAHALANOBIS = 15.0

# Professional Dark Mode Colors
COLOR_BG = '#121212'
COLOR_PANEL = '#1E1E1E'
COLOR_TEXT = '#FFFFFF'
COLOR_SUCCESS = '#2ECC71'
COLOR_ERROR = '#E74C3C'
COLOR_WARNING = '#F39C12'
COLOR_INFO = '#3498DB'

def adicionar_ruido(imagens: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0., 1.)

def calcular_mahalanobis_batch(vetores, mu, inv_sigma):
    diff = vetores - mu
    left = np.dot(diff, inv_sigma)
    dist_quadrada = np.sum(left * diff, axis=1)
    return np.sqrt(dist_quadrada)

def carregar_modelos():
    logger.info("Loading CNN and Mahalanobis profiles...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    ckpt.restore(tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))).expect_partial()
    mapa_ilhas = np.load(os.path.join("outputs", "mahalanobis_profiles.npz"), allow_pickle=True)
    return cnn, mapa_ilhas

def construir_banco_memoria_visual(x_train, y_train, cnn, size=10000):
    logger.info(f"Building Visual Memory Bank with {size} images...")
    idx = np.random.choice(len(x_train), size=size, replace=False)
    x_sample = x_train[idx]
    y_sample = y_train[idx]
    
    x_ruido = adicionar_ruido(x_sample, 0.6)
    
    batch_t = tf.convert_to_tensor(x_ruido, dtype=tf.float32)
    outputs = cnn(batch_t)
    features_128d = outputs["latent_features"].numpy()
    
    nn_index = NearestNeighbors(n_neighbors=15, algorithm='auto', metric='euclidean', n_jobs=-1)
    nn_index.fit(features_128d)
    
    return nn_index, features_128d, x_ruido, y_sample

def procurar_caso_heroi(x_test, y_test, cnn, mapa_ilhas, kdtree, y_memoria):
    logger.info("Searching for a Hero Case in the test set...")
    x_ruido = adicionar_ruido(x_test[:5000], 0.6)
    y_true = y_test[:5000]
    
    batch_size = 500
    for i in range(0, len(x_ruido), batch_size):
        batch = x_ruido[i : i + batch_size]
        batch_t = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_t)
        features_128d = outputs["latent_features"].numpy()
        probs = outputs["probabilities"].numpy()
        preds_cnn = np.argmax(probs, axis=1)
        
        todas_distancias = np.zeros((len(features_128d), 10))
        for d in range(10):
            mu = mapa_ilhas[str(d)].item()["mu"]
            inv_sigma = mapa_ilhas[str(d)].item()["inv_sigma"]
            todas_distancias[:, d] = calcular_mahalanobis_batch(features_128d, mu, inv_sigma)
        distancias_minimas = np.min(todas_distancias, axis=1)
        
        for j in range(len(batch)):
            real_label = y_true[i+j]
            pred_cnn = preds_cnn[j]
            dist_mah = distancias_minimas[j]
            
            if pred_cnn != real_label and dist_mah > LIMIAR_MAHALANOBIS:
                q_vec = features_128d[j].reshape(1, -1)
                dists, indices = kdtree.kneighbors(q_vec)
                neighbor_labels = y_memoria[indices[0]]
                
                counts = np.bincount(neighbor_labels, minlength=10)
                pred_knn = np.argmax(counts)
                
                if pred_knn == real_label:
                    logger.info(f"Hero Case found! Index {i+j} | Real: {real_label} | CNN: {pred_cnn} | k-NN: {pred_knn}")
                    return {
                        'imagem': batch[j],
                        'real_label': real_label,
                        'pred_cnn': pred_cnn,
                        'confianca_cnn': np.max(probs[j]),
                        'dist_mah': dist_mah,
                        'neighbor_indices': indices[0],
                        'neighbor_dists': dists[0],
                        'neighbor_labels': neighbor_labels,
                        'votes': counts
                    }
                    
    raise RuntimeError("No hero case found in these parameters!")

def plot_zona_1(ax, hero):
    ax.set_facecolor(COLOR_PANEL)
    ax.axis('off')
    
    ax.text(0.5, 0.94, "ZONE 1: THE CRISIS", ha='center', fontsize=16, fontweight='bold', color=COLOR_WARNING)
    
    # Push image slightly higher
    ax_img = ax.inset_axes([0.25, 0.50, 0.5, 0.35])
    ax_img.imshow(hero['imagem'][:,:,0], cmap='gray')
    ax_img.axis('off')
    ax_img.set_title("Input (Noise σ=0.6)", color=COLOR_TEXT, fontsize=12)
    
    # Adjust spacing to avoid overlap at the bottom
    text_y = 0.40
    ax.text(0.5, text_y, f"CNN Prediction: {hero['pred_cnn']}", ha='center', fontsize=20, fontweight='bold', color=COLOR_ERROR)
    text_y -= 0.07
    ax.text(0.5, text_y, f"Confidence: {hero['confianca_cnn']*100:.1f}%", ha='center', fontsize=14, color=COLOR_TEXT)
    text_y -= 0.10
    ax.text(0.5, text_y, "MAHALANOBIS ROUTING", ha='center', fontsize=12, color=COLOR_INFO, fontweight='bold')
    text_y -= 0.07
    ax.text(0.5, text_y, f"Distance: {hero['dist_mah']:.1f}", ha='center', fontsize=22, fontweight='bold', color=COLOR_ERROR)
    text_y -= 0.05
    ax.text(0.5, text_y, f"(Threshold = {LIMIAR_MAHALANOBIS})", ha='center', fontsize=12, color=COLOR_TEXT)
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc=COLOR_ERROR, ec="none", alpha=0.3)
    ax.text(0.5, 0.04, "REJECTED - ROUTE TO k-NN", ha='center', fontsize=14, fontweight='bold', color=COLOR_ERROR, bbox=bbox_props)

def plot_zona_2(ax_parent, hero, imagens_memoria, is_standalone=False):
    ax_parent.axis('off')
    if is_standalone:
        ax_parent.set_facecolor(COLOR_BG)
        ax_parent.text(0.5, 0.95, "ZONE 2: 128D LATENT SPACE (15 NEAREST MEMORIES)", ha='center', fontsize=16, fontweight='bold', color=COLOR_INFO)
        ax_parent.text(0.5, 0.88, "The agent extracts the 128D representation and retrieves past experiences with the shortest Euclidean distance.", ha='center', fontsize=12, color=COLOR_TEXT, alpha=0.8)
        # Increased hspace and top margin to prevent title/text overlapping
        gs_inner = gridspec.GridSpec(3, 5, top=0.75, bottom=0.08, left=0.05, right=0.95, wspace=0.1, hspace=1.2)
    else:
        ax_parent.text(0.5, 1.0, "ZONE 2: 128D LATENT SPACE (15 NEAREST MEMORIES)", ha='center', fontsize=16, fontweight='bold', color=COLOR_INFO)
        ax_parent.text(0.5, 0.95, "The agent extracts the 128D representation and retrieves past experiences with the shortest Euclidean distance.", ha='center', fontsize=12, color=COLOR_TEXT, alpha=0.8)
        gs_inner = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=ax_parent.get_subplotspec(), wspace=0.1, hspace=0.8)

    for i in range(15):
        row = i // 5
        col = i % 5
        ax_n = ax_parent.figure.add_subplot(gs_inner[row, col])
        
        idx_mem = hero['neighbor_indices'][i]
        dist = hero['neighbor_dists'][i]
        label = hero['neighbor_labels'][i]
        img_mem = imagens_memoria[idx_mem]
        
        ax_n.imshow(img_mem[:,:,0], cmap='gray')
        ax_n.axis('off')
        
        cor_label = COLOR_SUCCESS if label == hero['real_label'] else COLOR_ERROR
        
        ax_n.set_title(f"Label: {label}", color=cor_label, fontsize=14, fontweight='bold')
        # Pushed down slightly to give the title below it more room
        ax_n.text(0.5, -0.25, f"D: {dist:.1f}", ha='center', transform=ax_n.transAxes, color=COLOR_TEXT, fontsize=11)

def plot_zona_3(ax, hero):
    ax.set_facecolor(COLOR_PANEL)
    ax.axis('off')
    
    ax.text(0.5, 0.92, "ZONE 3: VOTE & ACTION", ha='center', fontsize=16, fontweight='bold', color=COLOR_SUCCESS)
    
    ax_bar = ax.inset_axes([0.1, 0.45, 0.8, 0.35])
    ax_bar.set_facecolor(COLOR_PANEL)
    
    digits = np.arange(10)
    votes = hero['votes']
    
    bar_colors = [COLOR_INFO] * 10
    pred_knn = np.argmax(votes)
    bar_colors[pred_knn] = COLOR_SUCCESS
    
    bars = ax_bar.bar(digits, votes, color=bar_colors, edgecolor='white', linewidth=1)
    ax_bar.set_xticks(digits)
    ax_bar.set_yticks(range(0, max(votes)+2, 2))
    ax_bar.tick_params(colors=COLOR_TEXT)
    for spine in ax_bar.spines.values():
        spine.set_color('#444444')
        
    ax_bar.set_title("Neighbor Votes (k=15)", color=COLOR_TEXT, fontsize=12)
    
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{int(h)}', ha='center', color=COLOR_TEXT, fontsize=10, fontweight='bold')

    text_y = 0.25
    ax.text(0.5, text_y, f"True Label: {hero['real_label']}", ha='center', fontsize=16, color=COLOR_TEXT)
    text_y -= 0.1
    ax.text(0.5, text_y, f"k-NN ACTION: {pred_knn}", ha='center', fontsize=26, fontweight='bold', color=COLOR_SUCCESS)
    text_y -= 0.1
    
    bbox_props_success = dict(boxstyle="round,pad=0.5", fc=COLOR_SUCCESS, ec="none", alpha=0.3)
    ax.text(0.5, 0.05, "CNN SUCCESSFULLY CORRECTED!", ha='center', fontsize=14, fontweight='bold', color=COLOR_SUCCESS, bbox=bbox_props_success)

def gerar_dashboards(hero, imagens_memoria, base_dir):
    plt.style.use('dark_background')
    
    # --- FULL DASHBOARD ---
    logger.info("Generating Full Dashboard...")
    fig = plt.figure(figsize=(22, 10), facecolor=COLOR_BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.15)
    fig.suptitle("128D EPISODIC MEMORY RETRIEVAL", fontsize=24, fontweight='bold', color=COLOR_TEXT, y=0.96)
    
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 2.5, 1.2])
    
    ax1 = fig.add_subplot(gs[0])
    plot_zona_1(ax1, hero)
    
    ax2 = fig.add_subplot(gs[1])
    # Tweak position for full dashboard
    gs_inner = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs[1], hspace=0.4, wspace=0.1)
    ax_center_title = fig.add_subplot(gs_inner[0, :])
    ax_center_title.axis('off')
    ax_center_title.text(0.5, 0.7, "ZONE 2: 128D LATENT SPACE (15 NEAREST MEMORIES)", ha='center', fontsize=16, fontweight='bold', color=COLOR_INFO)
    ax_center_title.text(0.5, 0.2, "The agent extracts the 128D representation and retrieves past experiences with the shortest Euclidean distance.", ha='center', fontsize=12, color=COLOR_TEXT, alpha=0.8)
    for i in range(15):
        row = (i // 5) + 1
        col = i % 5
        ax_n = fig.add_subplot(gs_inner[row, col])
        idx_mem = hero['neighbor_indices'][i]
        dist = hero['neighbor_dists'][i]
        label = hero['neighbor_labels'][i]
        img_mem = imagens_memoria[idx_mem]
        ax_n.imshow(img_mem[:,:,0], cmap='gray')
        ax_n.axis('off')
        cor_label = COLOR_SUCCESS if label == hero['real_label'] else COLOR_ERROR
        ax_n.set_title(f"Label: {label}", color=cor_label, fontsize=14, fontweight='bold')
        ax_n.text(0.5, -0.15, f"D: {dist:.1f}", ha='center', transform=ax_n.transAxes, color=COLOR_TEXT, fontsize=10)
    
    ax3 = fig.add_subplot(gs[2])
    plot_zona_3(ax3, hero)
    
    # Dividers
    line1 = plt.Line2D((0.28, 0.28), (0.05, 0.9), color='#444444', linewidth=2, linestyle='--')
    line2 = plt.Line2D((0.71, 0.71), (0.05, 0.9), color='#444444', linewidth=2, linestyle='--')
    fig.add_artist(line1)
    fig.add_artist(line2)

    full_path = os.path.join(base_dir, "episodic_memory_rescue_full.png")
    plt.savefig(full_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    # --- INDIVIDUAL ZONES ---
    logger.info("Generating Individual Zone Images...")
    
    # Zone 1
    fig1 = plt.figure(figsize=(5, 6), facecolor=COLOR_BG)
    ax_z1 = fig1.add_subplot(111)
    plot_zona_1(ax_z1, hero)
    plt.savefig(os.path.join(base_dir, "episodic_memory_rescue_zone1.png"), dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
    plt.close()
    
    # Zone 2
    fig2 = plt.figure(figsize=(10, 6), facecolor=COLOR_BG)
    ax_z2 = fig2.add_subplot(111)
    plot_zona_2(ax_z2, hero, imagens_memoria, is_standalone=True)
    plt.savefig(os.path.join(base_dir, "episodic_memory_rescue_zone2.png"), dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close()

    # Zone 3
    fig3 = plt.figure(figsize=(5, 6), facecolor=COLOR_BG)
    ax_z3 = fig3.add_subplot(111)
    plot_zona_3(ax_z3, hero)
    plt.savefig(os.path.join(base_dir, "episodic_memory_rescue_zone3.png"), dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
    plt.close()

    logger.info("[OK] All images generated successfully!")

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    os.makedirs("assets", exist_ok=True)
            
    cnn, mapa_ilhas = carregar_modelos()
    
    logger.info("Loading MNIST dataset...")
    x_train, y_train = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='train')
    x_train = x_train.astype(np.float32) / 255.0
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0
    
    kdtree, _, imagens_memoria, y_memoria = construir_banco_memoria_visual(x_train, y_train, cnn)
    
    hero = procurar_caso_heroi(x_test, y_test, cnn, mapa_ilhas, kdtree, y_memoria)
    
    gerar_dashboards(hero, imagens_memoria, "assets")

if __name__ == "__main__":
    main()
