"""Paper plot labels and visual encodings."""
from matplotlib.lines import Line2D

ALL=['haiku45','sonnet46','opus46','qwen3_vl_235b','gpt_oss_safeguard_120b']


FOCAL=ALL[2:]


MODEL_LABEL=dict(zip(ALL,['Haiku 4.5','Sonnet 4.6','Opus 4.6','Qwen3 VL 235B','GPT-Safe 120B']))


MODEL_COLOR=dict(zip(ALL,['#CC79A7','#A07819','#0072B2','#D55E00','#009E73']))


CATS=['baseline','subclinical','type1','type2']


CAT_LABEL=dict(zip(CATS,['Majority','Non-clinical Challenges','Protected (non-disability)','Protected (disability)']))


CAT_TICK=dict(zip(CATS,['Majority','Non-clinical\nChallenges','Protected\n(non-disability)','Protected\n(disability)']))


CAT_MARKER=dict(zip(CATS,['o','s','^','D']))


PHRASES=['explicit_brief','explicit_detailed','implicit_brief','implicit_detailed']


PHRASE_LABEL=dict(zip(PHRASES,['Explicit + brief','Explicit + detailed','Implicit + brief','Implicit + detailed']))


PHRASE_COLOR=dict(zip(PHRASES,['#6A3D9A','#6A3D9A','#606060','#606060']))


PHRASE_FACE={p:PHRASE_COLOR[p] if p.endswith('detailed') else 'white' for p in PHRASES}


def style(scope,r):
    color=PHRASE_COLOR[r.phrasing_id] if scope=='c' else MODEL_COLOR[r.model_id]
    marker=CAT_MARKER[r.category_id] if scope!='a' else 'o'
    return color,marker


def handles(scope):
    if scope=='a':
        top=[Line2D([],[],marker='o',color=MODEL_COLOR[m],ls='',ms=7,label=MODEL_LABEL[m]) for m in ALL]
        bottom=[]
    else:
        keys=FOCAL if scope=='b' else PHRASES
        colors=MODEL_COLOR if scope=='b' else PHRASE_COLOR
        names=MODEL_LABEL if scope=='b' else PHRASE_LABEL
        top=[Line2D([],[],marker='o',color=colors[k],ls='',ms=7,label=names[k],
            markerfacecolor=PHRASE_FACE[k] if scope=='c' else colors[k],
            markeredgewidth=1.3 if scope=='c' else 1) for k in keys]
        bottom=[Line2D([],[],marker=CAT_MARKER[c],color='.35',ls='',ms=7,label=CAT_LABEL[c]) for c in CATS]
    return top,bottom
