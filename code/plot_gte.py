"""Paper distance plots."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plot_style as style

TITLES = {'a': 'Response Shift by Model', 'b': 'Response Shift by Category and Model',
          'c': 'Response Shift by Category and Disclosure Phrasing',
          'violin': 'Response Shift by Model (Distribution)'}


STEMS = {'a': 'response_shift_by_model', 'b': 'response_shift_by_category_and_model',
         'c': 'response_shift_by_category_and_disclosure_phrasing',
         'violin': 'response_shift_by_model_violin'}


XLABEL = 'Cosine distance minus pooled rephrasing mean'


def decorate(ax, title, labels, limits):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.35, -.55)
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(MultipleLocator(.025 if limits[1]-limits[0] < .25 else .05))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel(XLABEL, labelpad=6)
    ax.set_title(title, fontsize=12, fontweight='semibold', pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=10)


def dot_figure(scope, table, limits):
    fig, ax = plt.subplots(figsize=(9.4, 3.0 if scope != 'a' else 2.8))
    fig.subplots_adjust(left=.20, right=.98 if scope == 'a' else .76, top=.93, bottom=.20)
    for row in table[table.scope == scope].itertuples():
        color = style.style(scope, row)[0]
        y = style.ALL.index(row.model_id) if scope == 'a' else style.CATS.index(row.category_id) + (
            [-.20, 0, .20][style.FOCAL.index(row.model_id)] if scope == 'b' else
            [-.27, -.09, .09, .27][style.PHRASES.index(row.phrasing_id)])
        ax.errorbar(row.mean, y, xerr=[[row.mean-row.low], [row.high-row.mean]], fmt='o',
            color=color, ecolor=color, capsize=3, ms=7 if scope == 'c' else 6, elinewidth=1.3,
            markerfacecolor=style.PHRASE_FACE[row.phrasing_id] if scope == 'c' else color,
            markeredgecolor=color if scope == 'c' else 'white', markeredgewidth=1.4 if scope == 'c' else .55,
            zorder=3)
    labels = [style.MODEL_LABEL[m] for m in style.ALL] if scope == 'a' else [style.CAT_TICK[c] for c in style.CATS]
    decorate(ax, TITLES[scope], labels, limits)
    if scope != 'a':
        fig.legend(handles=style.handles(scope)[0], loc='center left', bbox_to_anchor=(.78,.56),
            frameon=False, fontsize=10, handlelength=1.3, handletextpad=.55, labelspacing=.65, borderaxespad=0)
    return fig
