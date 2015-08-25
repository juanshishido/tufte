import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rc("savefig", dpi=200)
params = {#'figure.dpi' : 200,
          'figure.facecolor' : 'white',
          'axes.axisbelow' : True,
          
          'font.family' : 'serif',
          'font.serif' : 'Bitstream Vera Serif, New Century Schoolbook, Century Schoolbook L,\
                          Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman,\
                          Times, Palatino, Charter, serif',
                    
          'lines.antialiased' : True,
          
          'savefig.facecolor' : 'white'}

for (k, v) in params.iteritems():
    plt.rcParams[k] = v


def plot_style(ax, is_bar=False):

    if is_bar:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(0.75)

        ax.spines['bottom'].set_edgecolor('LightGray')
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(0.75)
        ax.spines['left'].set_linewidth(0.75)

        ax.spines['bottom'].set_edgecolor('#4B4B4B')
        ax.spines['left'].set_edgecolor('#4B4B4B')

    ax.tick_params(axis='both', top='off', bottom='off', left='off', right='off', colors='#4B4B4B', pad=10)

    ax.xaxis.label.set_color('#4B4B4B')
    ax.yaxis.label.set_color('#4B4B4B')


def range_frame(ax, x, y, fontsize):

    xmin = x.min()
    xmax = x.max()
    
    ymin = y.min()
    ymax = y.max()
    
    # Labels: x
    xlabels = [int(xl) for xl in ax.xaxis.get_majorticklocs() if xl > xmin and xl < xmax]
    xlabels = [xmin] + xlabels + [xmax]
    ax.set_xticks(xlabels)
    ax.set_xticklabels(xlabels, fontsize=fontsize)
    
    # Labels: y
    ylabels = [int(yl) for yl in ax.yaxis.get_majorticklocs() if yl > ymin and yl < ymax]
    ylabels = [ymin] + ylabels + [ymax]
    ax.set_yticks(ylabels)
    ax.set_yticklabels(ylabels, fontsize=fontsize)
    
    # Bounds: x
    xlower = xmin - ((xmax - xmin) * 0.05)
    xupper = xmax + ((xmax - xmin) * 0.05)
    
    # Bounds: y
    ylower = ymin - ((ymax - ymin) * 0.05)
    yupper = ymax + ((ymax - ymin) * 0.05)
    
    ax.set_xlim(xmin=xlower, xmax=xupper)
    ax.set_ylim(ymin=ylower, ymax=yupper)
    
    ax.spines['bottom'].set_bounds(xmin, xmax)
    ax.spines['left'].set_bounds(ymin, ymax)

    return ax


def auto_rotate_xticklabel(fig, ax):
    
    # Figure width (inches)
    figw = fig.get_figwidth()
    
    nticks = len(ax.xaxis.get_majorticklocs())
    
    # Spacing per tick (inches)
    tick_spacing = (figw / float(nticks))
    
    # Font size (pt)
    font_size = [v.get_fontsize() for v in ax.xaxis.get_majorticklabels()][0]
    
    # Font conversion (pt to in) [depends on the font, though]
    FONT_RATE = 0.01
    
    # Character width (inches)
    char_width = font_size * FONT_RATE
    
    # Maximum tick label width (inches)
    max_labelwidth = max([len(v.get_text()) for v in ax.xaxis.get_majorticklabels()]) * char_width
    
    if float(max_labelwidth) / tick_spacing >= 0.90:
        plt.xticks(rotation = 90)
    else:
        pass

    return fig, ax


def to_nparray(container):
    if type(container) in (list, pd.core.index.Int64Index, pd.Series):
        container = np.array(container)
    elif type(container) is np.ndarray:
        pass
    else:
        raise ValueError('Container must be: list, np.array, pd.core.index.Int64Index, or pd.Series')

    return container


def check_df(x, y, df):

    if isinstance(df, pd.DataFrame):
        if type(x) is str and type(y) is str:
            x = df[x]
            y = df[y]
        else:
            raise TypeError('x and y must be type str')
    else:
        if df is None:
            pass
        else:
            raise TypeError('df must be a pd.DataFrame')

    return (to_nparray(x), to_nparray(y))


def valid_x(x):

    if isinstance(x, pd.DataFrame):
        return True
    elif type(x) in (list, np.ndarray, pd.Series):
        return True
    else:
        return False


def scatter(x, y, df=None, figsize=(16, 8), marker='o', s=25, color='black', edgecolor='none', alpha=0.9, ticklabelsize=10):

    x, y = check_df(x, y, df)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, marker=marker, s=s, color=color, edgecolor=edgecolor, alpha=alpha)

    plot_style(ax, is_bar=False)

    ax = range_frame(ax, x, y, fontsize=ticklabelsize)

    return fig, ax


def line(x, y, df=None, figsize=(16, 8), linestyle='tufte', linewidth=1.0, color='black', alpha=0.9, ticklabelsize=10, markersize=10, **kwargs):

    x, y = check_df(x, y, df)

    fig, ax = plt.subplots(figsize=figsize)

    if linestyle == 'tufte':

        if len(kwargs) > 0:
            warnings.warn('Marker options are being ignored')

        marker = 'o'

        ax.plot(x, y, linestyle='-', linewidth=linewidth, color=color, alpha=alpha, zorder=1)
        ax.scatter(x, y, marker=marker, s=markersize*8, color='white', zorder=2)
        ax.scatter(x, y, marker=marker, s=markersize, color=color, zorder=3)
    else:
        ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha, markersize=markersize ** 0.5, **kwargs)

    plot_style(ax, is_bar=False)

    ax = range_frame(ax, x, y, fontsize=ticklabelsize)

    return fig, ax


def bar(position, height, df=None, label=None, align='center', color='LightGray', edgecolor='none', width=0.5, gridcolor='white'):

    position, height = check_df(position, height, df)

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.bar(position, height, align=align, color=color, edgecolor=edgecolor, width=width)

    plot_style(ax, is_bar=True)

    xmin = position.min()
    xmax = position.max()

    xlist = ax.xaxis.get_majorticklocs()

    if align is 'center':
        lower_buffer = 0.5
        upper_buffer = 0.5
    elif 'edge':
        lower_buffer = 0.25
        upper_buffer = width + 0.25

    xlist = [xl for xl in ax.xaxis.get_majorticklocs() if xl >= xmin and xl <= xmax]
    xlist = [xmin - lower_buffer] + xlist[1:-1] + [xmax + upper_buffer]

    for y in ax.yaxis.get_majorticklocs():
        ax.plot([xlist[0], xlist[-1]], [y, y], color=gridcolor, linewidth=1.25)

    ax.set_xlim(xmin=xlist[0], xmax=xlist[-1])

    if label is None:
        pass
    elif type(label) in (list, np.ndarray, pd.Series):
        label = np.array([str(lab) for lab in label])
        if len(label) == len(position):
            ax.set_xticks(position)
            ax.set_xticklabels(label)

            fig, ax = auto_rotate_xticklabel(fig, ax)
        else:
            raise ValueError('Labels must have the same first dimension as position and height')
    else:
        raise ValueError('Labels must be in: list, np.array, or pd.Series')
 
    return fig, ax


def bplot(x):

    if valid_x(x):

        fig, ax = plt.subplots(figsize=(16, 8))

        plot_style(ax, is_bar=False)

        ax.spines['bottom'].set_visible(False) 

        if isinstance(x, pd.DataFrame):

            i_pos = []

            for i, c in enumerate(x.columns):
                tdf = np.array(x[[c]])
                
                v000 = tdf.min()
                v025 = np.percentile(tdf, 25)
                v050 = np.median(tdf)
                v075 = np.percentile(tdf, 75)
                v100 = tdf.max()
                
                ax.plot([i, i], [v000, v025], color='black', linewidth=0.5)
                ax.plot([i, i], [v075, v100], color='black', linewidth=0.5)
                ax.scatter([i], [v050], color='black', s=5)

                i_pos.append(i)

            ax.set_xticks(i_pos)
            ax.set_xticklabels(x.columns)

        elif type(x) in (list, np.ndarray, pd.Series):
            x = to_nparray(x)
            
            v000 = x.min()
            v025 = np.percentile(x, 25)
            v050 = np.median(x)
            v075 = np.percentile(x, 75)
            v100 = x.max()

            ax.plot([0, 0], [v000, v025], color='black', linewidth=0.5)
            ax.plot([0, 0], [v075, v100], color='black', linewidth=0.5)
            ax.scatter([0], [v050], color='black', s=5)

            ax.axes.get_xaxis().set_visible(False)


    else:
        raise ValueError('x must be: list, np.array, pd.Series, or pd.DataFrame')
