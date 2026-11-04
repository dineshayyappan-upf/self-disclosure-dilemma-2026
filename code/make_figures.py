#!/usr/bin/env python3
"""Render the paper's three distance plots and six behavioral radars."""
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'reviewer_plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plot_gte
import plot_radars
import behavioral

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'outputs/figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':11,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    table = pd.read_csv(ROOT / 'data/gte/summary_centered_distance.csv')
    limits = (min(0, float(table.low.min())-.012), float(table.high.max())+.012)
    for scope in ['a','b','c']:
        fig = plot_gte.dot_figure(scope, table, limits)
        for ext in ['pdf','png']:
            fig.savefig(out / (plot_gte.STEMS[scope]+'.'+ext), dpi=300,
                        bbox_inches='tight', pad_inches=.06, facecolor='white')
        plt.close(fig)
    rates = pd.read_csv(ROOT / 'data/tables/condition_radars.csv')
    for tag in behavioral.TAG_ORDER:
        fig = plot_radars.draw_figure(rates, tag)
        for ext in ['pdf','png']:
            fig.savefig(out / (tag+'_condition_radars.'+ext), dpi=300, bbox_inches=None)
        plt.close(fig)
    print('Rendered three distance plots and six condition radars in outputs/figures/.')


if __name__ == '__main__':
    main()
