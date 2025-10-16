# picture.py
"""
Figures for the ILF3/PTBP1 case study.

Inputs (expected under `model_dir`):
  - top{K}_ILF3_PTBP1_disturb.csv   (K in {50,30,10})
      Required columns: Protein_A, Protein_B, disturb_score
  - top_ILF3_PTBP1_nodes_{K}.txt    (optional, for enrichment)
      One UniProt ID per line.

How to obtain the CSV/TXT:
  1) First run your predict script to produce the master table:
       {model_dir}/test_pairs_with_disturb_scores.csv
  2) Filter pairs where Protein_A or Protein_B is ILF3 (Q12906) or PTBP1 (P26599),
     sort by disturb_score desc, and take top-K to save as:
       top{K}_ILF3_PTBP1_disturb.csv
     Also write the node list of those top-K pairs to:
       top_ILF3_PTBP1_nodes_{K}.txt
  (This mirrors the dataset you used in your analysis; the plotting script itself
   DOES NOT generate these CSV/TXT files.)

Dependencies:
  pandas, numpy, matplotlib, networkx, seaborn, gseapy, (optional) mygene, openpyxl
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import gseapy as gp

# ----- EDIT ONLY THESE TWO PATHS -----
picture_dir = "..."       # folder to save figures/tables
model_dir   = "..."       # folder that contains the topK CSV/TXT files
# -------------------------------------

os.makedirs(picture_dir, exist_ok=True)

center_ids = ["Q12906", "P26599"]  # ILF3/PTBP1 UniProt IDs

for topk in [50, 30, 10]:
    # 1) Load disturbance score table
    csv_path = os.path.join(model_dir, f"top{topk}_ILF3_PTBP1_disturb.csv")
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found, skip Top{topk}.")
        continue
    df = pd.read_csv(csv_path)

    # === (A) Disturbance score histogram ===
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(
        df['disturb_score'], bins=20, color="#6baed6", edgecolor="k", alpha=0.7
    )
    plt.xlabel("Disturbance Score")
    plt.ylabel("Count")
    plt.title(f"Top-{topk} PPI Disturbance Score Distribution (ILF3/PTBP1 Case)")

    # highlight center proteins
    highlight_scores = []
    for a in center_ids:
        for _, row in df.iterrows():
            if a in (row["Protein_A"], row["Protein_B"]):
                highlight_scores.append(row["disturb_score"])
    highlight_scores = sorted(set(highlight_scores))
    for i, hs in enumerate(highlight_scores):
        plt.axvline(hs, color='red', linestyle='--', linewidth=1.5,
                    label='Center protein pair' if i == 0 else None)
    if highlight_scores:
        plt.legend(title="Red dashed line: Center protein pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, f"disturb_score_hist_highlight_top{topk}.png"))
    plt.close()

    # === (B) Disturbance network subgraph ===
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["Protein_A"], row["Protein_B"], weight=row["disturb_score"])

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["red" if n in center_ids else "#a6bddb" for n in G.nodes()]
    node_sizes = [800 if n in center_ids else 300 for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    width_scaled = [ (ew - min(edge_weights) + 1.0) * 3.0 for ew in edge_weights ]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=width_scaled, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"ILF3/PTBP1-centric PPI Subnetwork (Top-{topk})")

    from matplotlib.lines import Line2D
    plt.legend(
        [Line2D([0], [0], marker='o', color='w', label='ILF3/PTBP1',
                markerfacecolor='red', markersize=10)],
        ['ILF3/PTBP1 Center Node'],
        loc='lower left', frameon=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, f"ppi_subnetwork_highlight_top{topk}.png"))
    plt.close()

    # === (C) Enrichment analysis & bubble plot (optional; requires TXT node list) ===
    node_txt = os.path.join(model_dir, f"top_ILF3_PTBP1_nodes_{topk}.txt")
    if not os.path.exists(node_txt):
        print(f"{node_txt} not found, skip enrichment analysis.")
    else:
        with open(node_txt) as f:
            node_list = [x.strip() for x in f if x.strip()]
        gene_symbols = node_list
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            qres = mg.querymany(node_list, scopes="uniprot,ensemblprotein",
                                fields="symbol", species="human")
            gene_symbols = [x.get('symbol') for x in qres
                            if 'symbol' in x and x.get('notfound') is not True]
            print(f"Auto-converted to gene symbols: {len(gene_symbols)}")
        except Exception as e:
            print("Gene symbol conversion failed, using UniProt IDs.", e)

        try:
            enr = gp.enrichr(gene_list=gene_symbols,
                             gene_sets=['KEGG_2021_Human'],
                             organism='Human')
            results = enr.results.sort_values("Adjusted P-value").head(10)
            # bubble plot
            plt.figure(figsize=(7, 5))
            sizes = results['Overlap'].apply(lambda x: int(str(x).split('/')[0])) * 40
            plt.scatter(
                -results['Combined Score'],
                range(len(results)),
                s=sizes,
                c=-np.log10(results['P-value'])
            )
            plt.yticks(range(len(results)), results['Term'])
            plt.xlabel("-Combined Score")
            plt.colorbar(label="-log10(P-value)")
            plt.title(f"Top-{topk} KEGG Pathway Enrichment")
            plt.tight_layout()
            plt.savefig(os.path.join(picture_dir, f"top{topk}_enrich_bubble.png"), dpi=300)
            plt.close()
            print(f"Top-{topk} KEGG enrichment bubble plot generated")
        except Exception as e:
            print(f"Top-{topk} enrichment analysis or bubble plot failed:", e)

    # === (D) Save disturbance table (Excel) ===
    df.to_excel(os.path.join(picture_dir, f"top{topk}_disturb_table.xlsx"), index=False)
    print(f"Top{topk} visualizations and table saved.")

    # === (E) Disturbance heatmap ===
    try:
        proteins = sorted(set(df['Protein_A']).union(set(df['Protein_B'])))
        matrix = pd.DataFrame(np.nan, index=proteins, columns=proteins)
        for _, row in df.iterrows():
            matrix.loc[row['Protein_A'], row['Protein_B']] = row['disturb_score']
            matrix.loc[row['Protein_B'], row['Protein_A']] = row['disturb_score']  # symmetric
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix, cmap="coolwarm",
            center=np.nanmean(matrix.values),
            square=True, linewidths=.5, linecolor='gray',
            cbar_kws={"label": "Disturbance Score"}
        )
        plt.title(f"Top-{topk} Disturbance Score Heatmap (ILF3/PTBP1 Case)")
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, f"disturbance_heatmap_top{topk}.png"), dpi=300)
        plt.close()
        print(f"Top-{topk} disturbance heatmap generated")
    except Exception as e:
        print(f"Top-{topk} disturbance heatmap failed:", e)

print(f"\nAll Top-50/30/10 figures and tables exported to: {picture_dir}")
