#!/usr/bin/env python3
"""
make_figures_tables.py
======================
Regenerates all 9 paper figures and prints all result tables
from the paper_data/ folder produced by gather_paper_csvs.sh.

Usage
-----
    python make_figures_tables.py                  # uses ./paper_data/ auto-detected
    python make_figures_tables.py --data paper_data
    python make_figures_tables.py --data paper_data --out figs/

Output
------
    figs/fig1_architecture.pdf   -- skipped (hand-drawn diagram, not regenerated)
    figs/fig2_topologies.pdf     -- network visualisation + degree histograms
    figs/fig3_capability.pdf     -- Zipf capability bar chart
    figs/fig4_routing_score.pdf  -- composite score grouped bar chart
    figs/fig5_eff_latency.pdf    -- effective latency grouped bar chart
    figs/fig6_deadline_protocol.pdf -- deadline miss + protocol mismatch
    figs/fig7_scalability.pdf    -- decision time + latency vs scale
    figs/fig8_ablation.pdf       -- ablation: routing score / success / latency
    figs/fig9_xai.pdf            -- feature attribution horizontal bar

    tables_output.txt            -- all LaTeX-ready tables printed to file
"""

import argparse, os, sys, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── style ──────────────────────────────────────────────────────────────────────
COLORS = {
    'GloXRoute':    '#1F4E79',
    'ShortestPath': '#C00000',
    'Centralized':  '#7F7F7F',
    'SAW':          '#2E7D32',
    'TOPSIS':       '#6A0DAD',
    'BiRouter':     '#E65100',
    'Greedy':       '#4A4A4A',
    'DRL-OR-S':     '#00897B',
    'AgentNet':     '#AD1457',
}
METHOD_ORDER = ['GloXRoute','DRL-OR-S','AgentNet','BiRouter','ShortestPath','Centralized','SAW','TOPSIS','Greedy']
METHOD_MAP = {
    'HybridXAI_HeuristicCoord':'GloXRoute','SAW_MADM':'SAW','TOPSIS_MADM':'TOPSIS',
    'DecentralizedGreedy':'Greedy','BiRouterStyle':'BiRouter',
    'DRLOR_S':'DRL-OR-S','AgentNetStyle':'AgentNet',
}
TOPO_ORDER = ['BA','WS','ER']
COND_ORDER = ['normal','congestion','failure']
TOPO_LABEL = {'BA':'Barabási–Albert','WS':'Watts–Strogatz','ER':'Erdős–Rényi'}

plt.rcParams.update({
    'font.family':'sans-serif','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--',
})


# ── helpers ────────────────────────────────────────────────────────────────────

def holm(pvals):
    pvals = np.array(pvals, dtype=float)
    m = len(pvals); idx = np.argsort(pvals); adj = np.zeros(m); prev = 0
    for rank, i in enumerate(idx):
        a = min(1.0, (m-rank)*pvals[i]); a = max(a, prev); adj[i] = a; prev = a
    return adj

def load_main(data_dir):
    """Load and merge all 9 topology CSVs."""
    CONFIG_FILES = {
        ('BA','normal'):     'ba_normal_multiseed_all.csv',
        ('BA','congestion'): 'ba_congestion_multiseed_all.csv',
        ('BA','failure'):    'ba_failure_multiseed_all.csv',
        ('WS','normal'):     'ws_normal_multiseed_all.csv',
        ('WS','congestion'): 'ws_congestion_multiseed_all.csv',
        ('WS','failure'):    'ws_failure_multiseed_all.csv',
        ('ER','normal'):     'er_normal_multiseed_all.csv',
        ('ER','congestion'): 'er_congestion_multiseed_all.csv',
        ('ER','failure'):    'er_failure_multiseed_all.csv',
    }
    frames = []
    for (topo, cond), fname in CONFIG_FILES.items():
        path = Path(data_dir) / fname
        if not path.exists():
            print(f"  WARNING: missing {path}"); continue
        df = pd.read_csv(path)
        df['method'] = df['method'].map(METHOD_MAP).fillna(df['method'])
        df['topology'] = topo; df['condition'] = cond
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def grouped_bar(ax, data, configs, methods, colors, ylabel):

    import numpy as np

    x = np.arange(len(configs))

    width = 0.8 / len(methods)

    for i, method in enumerate(methods):

        values = [
            data[(t,c)][method]
            for t,c in configs
        ]

        ax.bar(
            x + i*width,
            values,
            width,
            label=method,
            color=colors[method]
        )


    ax.set_xticks(
        x + width*(len(methods)-1)/2
    )

    ax.set_xticklabels(
        [f"{t}-{c}" for t,c in configs],
        rotation=45,
        ha='right'
    )

    ax.set_ylabel(ylabel)

def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.savefig(path.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – topology visualisation
# ══════════════════════════════════════════════════════════════════════════════
def fig2_topologies(out_dir):
    try: import networkx as nx
    except ImportError: print("  skipping fig2 — networkx not installed"); return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    configs = [
        ('Barabási–Albert\n$|E|=291$, $\\bar{k}=5.82$', nx.barabasi_albert_graph(100,3,seed=42)),
        ('Watts–Strogatz\n$|E|=300$, $\\bar{k}=6.00$',  nx.connected_watts_strogatz_graph(100,6,0.3,seed=42)),
        ('Erdős–Rényi\n$|E|=305$, $\\bar{k}=6.10$',     nx.erdos_renyi_graph(100,0.062,seed=42)),
    ]
    cmap = plt.cm.viridis
    for col, (title, G) in enumerate(configs):
        ax = axes[0, col]
        degrees = dict(G.degree())
        pos = nx.spring_layout(G, seed=42)
        node_colors = [cmap(degrees[n] / max(degrees.values())) for n in G.nodes()]
        node_sizes  = [20 + degrees[n]*8 for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
        ax.set_title(title, fontsize=11, fontweight='bold'); ax.axis('off')
        ax_h = axes[1, col]
        deg_vals = [d for _, d in G.degree()]
        ax_h.hist(deg_vals, bins=15, color='#1F4E79', edgecolor='white', linewidth=0.4)
        ax_h.set_yscale('log'); ax_h.set_xlabel('degree'); ax_h.set_ylabel('count' if col==0 else '')
        mean_k = np.mean(deg_vals)
        ax_h.axvline(mean_k, color='red', lw=1, linestyle='--', alpha=0.7)
    fig.tight_layout(pad=1.5)
    save(fig, str(Path(out_dir)/'fig2_topologies.pdf'))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – capability distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig3_capability(out_dir):
    caps =  ['search','reasoning','coding','math','planning','translation','vision','storage']
    counts = [52, 19, 10, 7, 5, 3, 2, 2]
    bar_colors = ['#C00000'] + ['#1F4E79']*7
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(caps, counts, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(cnt), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('agents providing capability', fontsize=12)
    ax.set_title('Zipf capability assignment ($\\alpha=1.5$), 100 agents', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 62); ax.tick_params(axis='x', labelsize=10)
    fig.tight_layout()
    save(fig, str(Path(out_dir)/'fig3_capability.pdf'))


# ══════════════════════════════════════════════════════════════════════════════
# Figures 4 & 5 – composite score and effective latency
# ══════════════════════════════════════════════════════════════════════════════
def fig4_routing_score(all_df, out_dir):

    configs = [(t, c) for t in TOPO_ORDER for c in COND_ORDER]

    # Calculate mean routing score
    means = (
        all_df
        .groupby(['topology', 'condition', 'method'])['routing_score']
        .mean()
    )

    print("\n🚀 Fig4 dataframe shape:", all_df.shape)

    print("\n🚀 Available routing score statistics:")
    print(all_df['routing_score'].describe())

    print("\n🚀 Grouped means:")
    print(means)


    # Build plotting dictionary
    data = {}

    for t, c in configs:

        data[(t, c)] = {}

        for m in METHOD_ORDER:

            value = means.get((t, c, m), np.nan)

            data[(t, c)][m] = value


    print("\n🚀 Final plotting data:")
    for k, v in data.items():
        print(k, v)


    # Check if everything is NaN
    values = [
        v
        for config in data.values()
        for v in config.values()
        if not np.isnan(v)
    ]

    if len(values) == 0:
        raise ValueError(
            "Fig4 has no valid routing_score values. "
            "Check topology/condition/method names."
        )


    fig, ax = plt.subplots(figsize=(13, 5))


    # Correct grouped bar input
    grouped_bar(
        ax,
        data,
        configs,
        METHOD_ORDER,
        COLORS,
        'composite routing score'
    )


    # Legend
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=COLORS[m],
            label=m
        )
        for m in METHOD_ORDER
    ]

    ax.legend(
        handles=handles,
        ncol=min(9, len(METHOD_ORDER)),
        fontsize=8,
        loc='upper right',
        framealpha=0.9,
        edgecolor='0.8'
    )


    # Automatic y-axis
    ymin = min(values)
    ymax = max(values)

    margin = (ymax - ymin) * 0.15

    if margin == 0:
        margin = ymax * 0.1

    ax.set_ylim(
        max(0, ymin - margin),
        ymax + margin
    )


    ax.set_ylabel("Composite routing score")

    fig.tight_layout()


    save(
        fig,
        str(Path(out_dir) / 'fig4_routing_score.pdf')
    )

    plt.close(fig)

    print(
        "✅ Fig4 saved:",
        str(Path(out_dir) / 'fig4_routing_score.pdf')
    )

def fig5_eff_latency(all_df, out_dir):
    configs = [(t,c) for t in TOPO_ORDER for c in COND_ORDER]
    means = all_df.groupby(['topology','condition','method'])['avg_effective_latency'].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    handles = [plt.Rectangle((0,0),1,1,color=COLORS[m],label=m) for m in METHOD_ORDER]
    n_methods = len(METHOD_ORDER); width = 0.8/n_methods; x = np.arange(len(configs))
    for j, m in enumerate(METHOD_ORDER):
        vals = [means.get((t,c,m), np.nan) for t,c in configs]
        ax.bar(x+j*width-0.4+width/2, vals, width=width*0.9,
               color=COLORS[m], label=m, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n{c}' for t,c in configs], fontsize=9)
    ax.set_ylabel('effective latency'); ax.set_xlim(-0.5, len(configs)-0.5)
    for i in [2.5, 5.5]: ax.axvline(i, color='0.6', lw=0.8, linestyle=':')
    ax.legend(handles=handles, ncol=9, fontsize=8, loc='upper left',
              framealpha=0.9, edgecolor='0.8')
    fig.tight_layout()
    save(fig, str(Path(out_dir)/'fig5_eff_latency.pdf'))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – deadline miss + protocol mismatch
# ══════════════════════════════════════════════════════════════════════════════
def fig6_deadline_protocol(all_df, out_dir):
    # aggregate across all configs (pooled mean per method)
    order = [m for m in METHOD_ORDER if m in set(all_df['method'])]
    dl = all_df.groupby('method')['deadline_miss_rate'].mean()[order]
    pm = all_df.groupby('method')['protocol_mismatch_rate'].mean()[order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, ylabel, title in [
        (axes[0], dl, 'fraction of tasks', 'Deadline miss rate'),
        (axes[1], pm, 'fraction of hops',  'Protocol mismatch rate'),
    ]:
        bar_colors = [COLORS[m] for m in order]
        bars = ax.bar(order, data.values, color=bar_colors, edgecolor='white', zorder=3)
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
        ax.set_xticklabels(order, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0, data.max()*1.15)
    fig.tight_layout()
    save(fig, str(Path(out_dir)/'fig6_deadline_protocol.pdf'))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 – scalability
# ══════════════════════════════════════════════════════════════════════════════
def fig7_scalability(data_dir, out_dir):
    sc_path = Path(data_dir)/'scalability_summary.csv'
    if not sc_path.exists(): print("  WARNING: scalability_summary.csv not found"); return
    sc = pd.read_csv(sc_path)
    sc['method'] = sc['method'].map(METHOD_MAP).fillna(sc['method'])
    plot_methods = ['GloXRoute','ShortestPath','Greedy']
    plot_colors  = {'GloXRoute':'#1F4E79','ShortestPath':'#C00000','Greedy':'#4A4A4A'}
    plot_markers = {'GloXRoute':'o','ShortestPath':'s','Greedy':'^'}
    agents = sorted(sc['num_agents'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # (a) decision time – log scale
    ax = axes[0]
    for m in plot_methods:
        sub = sc[sc['method']==m].set_index('num_agents')
        y = sub['avg_decision_ms_mean'].reindex(agents)
        ci = sub['decision_time_ci95'].reindex(agents)
        ax.errorbar(agents, y, yerr=ci, label=m, color=plot_colors[m],
                    marker=plot_markers[m], linewidth=2, markersize=6, capsize=3)
    # annotate speedup
    g500 = sc[(sc['method']=='GloXRoute')&(sc['num_agents']==500)]['avg_decision_ms_mean'].values[0]
    s500 = sc[(sc['method']=='ShortestPath')&(sc['num_agents']==500)]['avg_decision_ms_mean'].values[0]
    ax.annotate(f'{s500/g500:.1f}×\nfaster', xy=(500, g500), xytext=(380, g500*4),
                arrowprops=dict(arrowstyle='->', color='#C00000'), color='#C00000', fontsize=10, fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(agents); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('number of agents'); ax.set_ylabel('decision time (ms)')
    ax.set_title('(a) Per-decision computation time', fontweight='bold')
    ax.legend(fontsize=10)
    # (b) routing quality
    ax = axes[1]
    for m in plot_methods:
        sub = sc[sc['method']==m].set_index('num_agents')
        y = sub['avg_eff_latency_mean'].reindex(agents)
        ci = sub['avg_eff_latency_std'].reindex(agents) * 1.96 / np.sqrt(10)
        ax.errorbar(agents, y, yerr=ci, label=m, color=plot_colors[m],
                    marker=plot_markers[m], linewidth=2, markersize=6, capsize=3)
    ax.set_xscale('log')
    ax.set_xticks(agents); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('number of agents'); ax.set_ylabel('effective latency')
    ax.set_title('(b) Routing quality vs. scale', fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, str(Path(out_dir)/'fig7_scalability.pdf'))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 – ablation
# ══════════════════════════════════════════════════════════════════════════════
# def fig8_ablation(data_dir, out_dir):
#     abl_path = Path(data_dir)/'ablation_all.csv'
#     if not abl_path.exists(): print("  WARNING: ablation_all.csv not found"); return
#     abl = pd.read_csv(abl_path)
#     VARIANT_LABEL = {
#         'GloXRoute_Full':        'Full',
#         'Ablation_NoGlobal':     '− global\ncoord.',
#         'Ablation_NoCapability': '− capability',
#         'Ablation_NoLoad':       '− load',
#         'Ablation_NoProtocol':   '− protocol',
#         'Ablation_NoReliability':'− reliability',
#     }
#     ORDER = list(VARIANT_LABEL.keys())
#     means = abl.groupby('method')[['routing_score','success_rate','avg_effective_latency']].mean()
#     stds  = abl.groupby('method')[['routing_score','success_rate','avg_effective_latency']].std(ddof=1)
#     labels = [VARIANT_LABEL[v] for v in ORDER]
#     highlight = [v == 'Ablation_NoGlobal' for v in ORDER]
#     bar_color = ['#C00000' if h else '#1F4E79' for h in highlight]
#     alpha_val = [1.0 if h else 0.6 for h in highlight]
#     fig, axes = plt.subplots(1, 3, figsize=(13, 5))
#     for ax, metric, ylabel, title in [
#         (axes[0], 'routing_score',          'score',   'Routing score'),
#         (axes[1], 'success_rate',           'fraction','Success rate'),
#         (axes[2], 'avg_effective_latency',  'latency', 'Effective latency'),
#     ]:
#         vals = means.reindex(ORDER)[metric].values
#         errs = stds.reindex(ORDER)[metric].values
#         bars = ax.bar(labels, vals, color=bar_color, alpha=1.0,
#                       yerr=errs, capsize=4, edgecolor='white', zorder=3)
#         for bar, h, a in zip(bars, highlight, alpha_val):
#             bar.set_alpha(1.0 if h else 0.55)
#         ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
#         ax.tick_params(axis='x', labelsize=9)
#     fig.tight_layout()
#     save(fig, str(Path(out_dir)/'fig8_ablation.pdf'))

def fig8_ablation(data_dir, out_dir):

    abl_path = Path(data_dir) / 'ablation_all.csv'

    if not abl_path.exists():
        print("  WARNING: ablation_all.csv not found")
        return


    abl = pd.read_csv(abl_path)


    VARIANT_LABEL = {
        'GloXRoute_Full':             'Full',
        'Ablation_NoGlobal':          '− global\ncoord.',
        'Ablation_NoCapability':      '− capability',
        'Ablation_NoLoad':            '− load',
        'Ablation_NoProtocol':        '− protocol',
        'Ablation_NoReliability':     '− reliability',
        'Ablation_NoDeadlineBudget':  '− deadline\nbudget',
        'Ablation_NoCostTerm':        '− cost\nterm',
    }

    ORDER = [v for v in VARIANT_LABEL if v in set(abl['method'])]


    means = (
        abl
        .groupby('method')[
            [
                'routing_score',
                'success_rate',
                'avg_effective_latency'
            ]
        ]
        .mean()
    )


    stds = (
        abl
        .groupby('method')[
            [
                'routing_score',
                'success_rate',
                'avg_effective_latency'
            ]
        ]
        .std(ddof=1)
    )


    labels = [
        VARIANT_LABEL[v]
        for v in ORDER
    ]


    # Highlight global coordinator removal
    bar_colors = [
        '#C00000' if v == 'Ablation_NoGlobal'
        else '#1F4E79'
        for v in ORDER
    ]


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.8)
    )


    metrics = [
        (
            axes[0],
            'routing_score',
            'score',
            'Routing score'
        ),
        (
            axes[1],
            'success_rate',
            'fraction',
            'Success rate'
        ),
        (
            axes[2],
            'avg_effective_latency',
            'latency',
            'Effective latency'
        ),
    ]


    x = np.arange(len(labels))


    for ax, metric, ylabel, title in metrics:

        vals = (
            means
            .reindex(ORDER)[metric]
            .values
        )

        errs = (
            stds
            .reindex(ORDER)[metric]
            .values
        )


        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=4,
            color=bar_colors,
            edgecolor='white',
            linewidth=0.6,
            zorder=3
        )


        # Make Full / highlighted variant obvious
        for idx, bar in enumerate(bars):

            if ORDER[idx] != 'Ablation_NoGlobal':
                bar.set_alpha(0.65)


        ax.set_title(
            title,
            fontsize=13,
            fontweight='bold'
        )

        ax.set_ylabel(
            ylabel,
            fontsize=11
        )


        ax.set_xticks(x)

        ax.set_xticklabels(
            labels,
            rotation=35,
            ha='right',
            fontsize=9
        )


        ax.grid(
            axis='y',
            alpha=0.3,
            linestyle='--'
        )


        ax.set_axisbelow(True)



    # More room for labels
    fig.subplots_adjust(
        bottom=0.30,
        wspace=0.30
    )


    fig.suptitle(
        'GloXRoute component ablation analysis',
        fontsize=15,
        fontweight='bold',
        y=1.02
    )


    save(
        fig,
        str(Path(out_dir) / 'fig8_ablation.pdf')
    )

# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 – XAI feature attribution
# ══════════════════════════════════════════════════════════════════════════════
def fig9_xai(data_dir, out_dir):
    fc_path = Path(data_dir)/'feature_consistency.csv'
    if not fc_path.exists(): print("  WARNING: feature_consistency.csv not found"); return
    fc = pd.read_csv(fc_path).sort_values('mean_contribution')
    colors = ['#C00000' if v < 0 else '#1F4E79' for v in fc['mean_contribution']]
    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(
        fc['feature'],
        fc['mean_contribution'],
        xerr=fc['std_contribution'],
        color=colors,
        capsize=4,
        edgecolor='white',
        zorder=3
    )

    # show zero/small contributions
    for i, value in enumerate(fc['mean_contribution']):

        if abs(value) < 0.01:
            ax.scatter(
                0,
                i,
                color='#1F4E79',
                s=45,
                zorder=5
            )
    
    ax.axvline(0, color='0.3', linewidth=1.0)
    ax.set_xlabel('mean weighted contribution to decision')
    ax.set_title('Feature attribution (1,426 decisions)', fontweight='bold')
    fig.tight_layout()
    save(fig, str(Path(out_dir)/'fig9_xai.pdf'))


# ══════════════════════════════════════════════════════════════════════════════
# Tables
# ══════════════════════════════════════════════════════════════════════════════
def print_tables(all_df, data_dir, out_file):
    lines = []
    def p(s=''): lines.append(s)
    M = METHOD_ORDER

    # helper
    def wilcoxon_holm(all_df, metric, m1='GloXRoute', m2='ShortestPath', alt='less'):
        rows = []
        for topo in TOPO_ORDER:
            for cond in COND_ORDER:
                sub = all_df[(all_df['topology']==topo)&(all_df['condition']==cond)]
                a = sub[sub['method']==m1][metric].values
                b = sub[sub['method']==m2][metric].values
                if len(a) < 2: continue
                _, pv = stats.wilcoxon(a, b, alternative=alt)
                diff = a - b; d = diff.mean()/diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
                rows.append({'topo':topo,'cond':cond,'G':a.mean(),'S':b.mean(),
                             'red':(1-a.mean()/b.mean())*100,'p_raw':pv,'d':d})
        df = pd.DataFrame(rows)
        df['p_holm'] = holm(df['p_raw'].values)
        return df

    # ── Table 7: Composite ────────────────────────────────────────────
    p("="*70); p("TABLE 7: COMPOSITE ROUTING SCORE"); p("="*70)
    pivot = all_df.groupby(['topology','condition','method'])['routing_score'].mean().unstack('method')
    pivot = pivot[[m for m in M if m in pivot.columns]].round(4)
    p(pivot.to_string())
    p()
    p("Wilcoxon GloXRoute vs ShortestPath (composite):")
    t7p = wilcoxon_holm(all_df,'routing_score',alt='greater')
    for _,r in t7p.iterrows():
        p(f"  {r.topo}/{r.cond:10s}  G={r.G:.4f}  S={r.S:.4f}  p_raw={r.p_raw:.2e}  p_holm={r.p_holm:.2e}")

    # ── Table 8: Deadline miss ────────────────────────────────────────
    p(); p("="*70); p("TABLE 8: DEADLINE MISS RATE (GloXRoute vs ShortestPath)"); p("="*70)
    t8 = wilcoxon_holm(all_df,'deadline_miss_rate')
    for _,r in t8.iterrows():
        p(f"  {r.topo}/{r.cond:10s}  G={r.G:.4f}  S={r.S:.4f}  "
          f"red={r.red:.0f}%  p_raw={r.p_raw:.2e}  p_holm={r.p_holm:.2e}  d={r.d:.3f}")
    p(f"\n  Holm max-p: {t8['p_holm'].max():.2e}")
    p(f"  |d| range:  {t8['d'].abs().min():.2f} – {t8['d'].abs().max():.2f}")

    # ── Table 8b: GloXRoute vs DRL-OR-S ──────────────────────────────
    p(); p("="*70); p("TABLE 8b: GloXRoute vs DRL-OR-S"); p("="*70)
    t8b_comp = wilcoxon_holm(all_df,'routing_score','GloXRoute','DRL-OR-S',alt='greater')
    t8b_dead = wilcoxon_holm(all_df,'deadline_miss_rate','GloXRoute','DRL-OR-S',alt='less')
    p(f"  {'Config':15s}  {'G composite':12s}  {'D composite':12s}  {'p_holm':10s}  {'G deadline':12s}  {'D deadline':12s}  {'p_holm':10s}")
    for (_,rc), (_,rd) in zip(t8b_comp.iterrows(), t8b_dead.iterrows()):
        p(f"  {rc.topo}/{rc.cond:10s}  {rc.G:.4f}        {rc.S:.4f}        {rc.p_holm:.2e}    {rd.G:.4f}        {rd.S:.4f}        {rd.p_holm:.2e}")

    # ── Table 6: BA/congestion full ───────────────────────────────────
    p(); p("="*70); p("TABLE 6: BA/CONGESTION — FULL METRICS (mean ± SD, 20 seeds)"); p("="*70)
    COLS = ['success_rate','avg_latency','avg_effective_latency','avg_hops',
            'avg_overhead','avg_path_stretch','protocol_mismatch_rate',
            'deadline_miss_rate','avg_decision_time_ms']
    sub = all_df[(all_df['topology']=='BA')&(all_df['condition']=='congestion')]
    mn = sub.groupby('method')[COLS].mean().round(4)
    sd = sub.groupby('method')[COLS].std(ddof=1).round(4)
    for m in [x for x in M if x in mn.index]:
        p(f"  {m:22s}")
        for c in COLS:
            p(f"    {c:30s}  {mn.loc[m,c]:.4f} ± {sd.loc[m,c]:.4f}")

    # ── Table 9: Effective latency ────────────────────────────────────
    p(); p("="*70); p("TABLE 9: EFFECTIVE LATENCY (means)"); p("="*70)
    t9 = all_df.groupby(['topology','condition','method'])['avg_effective_latency'].mean().unstack('method')
    t9 = t9[[m for m in M if m in t9.columns]].round(3)
    p(t9.to_string())

    # ── Table 10: Overhead ────────────────────────────────────────────
    p(); p("="*70); p("TABLE 10: CONTROL OVERHEAD (msgs/task)"); p("="*70)
    t10 = all_df.groupby(['topology','condition','method'])['avg_overhead'].mean().unstack('method')
    t10 = t10[[m for m in M if m in t10.columns]].round(3)
    p(t10.to_string())

    # ── Table 11: Ablation ────────────────────────────────────────────
    abl_path = Path(data_dir)/'ablation_all.csv'
    if abl_path.exists():
        p(); p("="*70); p("TABLE 11: ABLATION (BA topology, 20 seeds)"); p("="*70)
        abl = pd.read_csv(abl_path)
        VARIANT_ORDER = ['GloXRoute_Full','Ablation_NoGlobal','Ablation_NoCapability',
                         'Ablation_NoLoad','Ablation_NoProtocol','Ablation_NoReliability',
                         'Ablation_NoDeadlineBudget','Ablation_NoCostTerm']
        LABEL = {'GloXRoute_Full':'GloXRoute (full)','Ablation_NoGlobal':'− global coordinator',
                 'Ablation_NoCapability':'− capability','Ablation_NoLoad':'− load',
                 'Ablation_NoProtocol':'− protocol','Ablation_NoReliability':'− reliability',
                 'Ablation_NoDeadlineBudget':'− deadline budget','Ablation_NoCostTerm':'− cost term'}
        COLS2 = ['success_rate','avg_effective_latency','protocol_mismatch_rate','routing_score']
        mn2 = abl.groupby('method')[COLS2].mean().round(4)
        full_score = mn2.loc['GloXRoute_Full','routing_score']
        p(f"  {'Variant':30s}  {'Succ':6s}  {'EffLat':8s}  {'Proto':6s}  {'Score':6s}  {'ΔScore':8s}")
        for v in VARIANT_ORDER:
            if v not in mn2.index: continue
            r = mn2.loc[v]
            delta = r.routing_score - full_score
            p(f"  {LABEL.get(v,v):30s}  {r.success_rate:.4f}  {r.avg_effective_latency:.4f}  "
              f"{r.protocol_mismatch_rate:.4f}  {r.routing_score:.4f}  {delta:+.4f}")

    # ── Table 12: XAI ─────────────────────────────────────────────────
    xs_path = Path(data_dir)/'xai_summary.csv'
    fc_path = Path(data_dir)/'feature_consistency.csv'
    cf_path = Path(data_dir)/'counterfactuals.csv'
    if xs_path.exists() and fc_path.exists() and cf_path.exists():
        p(); p("="*70); p("TABLE 12: XAI EVALUATION"); p("="*70)
        xs = pd.read_csv(xs_path).set_index('Unnamed: 0')['0']
        cf = pd.read_csv(cf_path)
        fc = pd.read_csv(fc_path)
        p(f"  Fidelity:                         {float(xs['fidelity']):.3f}")
        p(f"  Decisions checked:                {int(float(xs['decisions_checked']))}")
        p(f"  Counterfactual delta (median):    {cf['delta_load_to_flip'].median():.3f}")
        p(f"  Counterfactual delta (mean):      {cf['delta_load_to_flip'].mean():.3f}")
        p(f"  Counterfactual delta (min):       {cf['delta_load_to_flip'].min():.3f}")
        p(); p(f"  {'Feature':15s}  {'Weight':6s}  {'Mean contrib':12s}  {'SD':8s}")
        for _, row in fc.iterrows():
            p(f"  {row['feature']:15s}  {row['weight']:+5.1f}   {row['mean_contribution']:+8.4f}        {row['std_contribution']:.4f}")

    # ── Table 13: Scalability ─────────────────────────────────────────
    sc_path = Path(data_dir)/'scalability_summary.csv'
    if sc_path.exists():
        p(); p("="*70); p("TABLE 13: SCALABILITY"); p("="*70)
        sc = pd.read_csv(sc_path)
        sc['method'] = sc['method'].map(METHOD_MAP).fillna(sc['method'])
        for m in ['GloXRoute','ShortestPath','Greedy']:
            sub = sc[sc['method']==m].sort_values('num_agents')
            p(f"  {m}:")
            for _, r in sub.iterrows():
                p(f"    {int(r.num_agents):4d} agents: dec={r.avg_decision_ms_mean:.4f}ms  "
                  f"eff_lat={r.avg_eff_latency_mean:.3f}")
        g50  = sc[(sc['method']=='GloXRoute')&(sc['num_agents']==50)]['avg_decision_ms_mean'].values[0]
        g500 = sc[(sc['method']=='GloXRoute')&(sc['num_agents']==500)]['avg_decision_ms_mean'].values[0]
        s50  = sc[(sc['method']=='ShortestPath')&(sc['num_agents']==50)]['avg_decision_ms_mean'].values[0]
        s500 = sc[(sc['method']=='ShortestPath')&(sc['num_agents']==500)]['avg_decision_ms_mean'].values[0]
        p(f"\n  GloXRoute growth 50→500: {g500/g50:.1f}×")
        p(f"  ShortestPath growth 50→500: {s500/s50:.1f}×")
        p(f"  Speedup at 500: {s500/g500:.1f}×")

    out = '\n'.join(lines)
    with open(out_file, 'w') as f: f.write(out)
    print(out)
    print(f"\n  Tables written to {out_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='Regenerate GloXRoute paper figures and tables')
    ap.add_argument('--data', default='paper_data', help='Folder with gathered CSVs (default: paper_data)')
    ap.add_argument('--out',  default='figs',       help='Output folder for figures (default: figs)')
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        sys.exit(f"ERROR: data folder not found: {data_dir}\n"
                 f"Run gather_paper_csvs.sh first.")

    print(f"Data:   {data_dir}")
    print(f"Figs:   {out_dir}")
    print()

    print("Loading main matrix...")
    all_df = load_main(data_dir)
    print(f"  {len(all_df)} rows, {all_df['seed'].nunique()} seeds, "
          f"{all_df['method'].nunique()} methods\n")

    print("Generating figures...")
    fig2_topologies(out_dir)
    fig3_capability(out_dir)
    fig4_routing_score(all_df, out_dir)
    fig5_eff_latency(all_df, out_dir)
    fig6_deadline_protocol(all_df, out_dir)
    fig7_scalability(data_dir, out_dir)
    fig8_ablation(data_dir, out_dir)
    fig9_xai(data_dir, out_dir)
    print()

    print("Computing tables...")
    print_tables(all_df, data_dir, str(out_dir/'tables_output.txt'))

    print(f"\nAll done. Figures and tables in: {out_dir}/")
    print("Note: fig1_architecture.pdf is a hand-drawn diagram — not regenerated.")
    print("Copy fig1_architecture.pdf from your uploads into figs/ manually.")


if __name__ == '__main__':
    main()