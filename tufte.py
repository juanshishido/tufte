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
        # No top, left, or right spines
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Spine linewidth
        ax.spines['bottom'].set_linewidth(0.75)

        # Spine color
        ax.spines['bottom'].set_edgecolor('LightGray')
    else:
        # No top or right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Spine linewidth
        ax.spines['bottom'].set_linewidth(0.75)
        ax.spines['left'].set_linewidth(0.75)

        # Spine color
        ax.spines['bottom'].set_edgecolor('#4B4B4B')
        ax.spines['left'].set_edgecolor('#4B4B4B')

    # No ticks marks, ticklabel color, pad
    ax.tick_params(axis='both', top='off', bottom='off', left='off', right='off', colors='#4B4B4B', pad=10)

    # Label color
    ax.xaxis.label.set_color('#4B4B4B')
    ax.yaxis.label.set_color('#4B4B4B')


def range_frame(ax, x, y, fontsize):
    """
    SCALE BOUNDS TO FIGURE
    ALSO:   WHAT HAPPENS WHEN THE MIN OR MAX VALUE IS CLOSE TO AN
            ACTUAL TICK VALUE? E.g., 1800 and 1801 -> OVERLAP?
    """

    # Range: x
    xmin = x.min()
    xmax = x.max()
    
    # Range: y
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
    
    # Axis limits
    ax.set_xlim(xmin=xlower, xmax=xupper)
    ax.set_ylim(ymin=ylower, ymax=yupper)
    
    # Faux axes lines
    ax.spines['bottom'].set_bounds(xmin, xmax)
    ax.spines['left'].set_bounds(ymin, ymax)

    return ax


def auto_rotate_xticklabel(fig, ax):
    
    # Figure width (inches)
    figw = fig.get_figwidth()
    
    # Number of ticks
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
    if type(container) in (list, pd.Series):
        container = np.array(container)
    elif type(container) is np.ndarray:
        pass
    else:
        raise ValueError('Container must be: list, np.array, or pd.Series')

    return container


def check_df(x, y, df):

    """
    A function to check whether a pd.DataFrame with column names has been passed in.
    If not, can also take type: list, np.array, pd.Series.
    Will return np.array() versions of x and y.
    """

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


def check_position(df, p):
    if isinstance(df, pd.DataFrame):
        if p is None:
            return None
        else:
            if type(p) is str:
                p = df[p]
            else:
                raise TypeError('p must be type str')
    else:
        if p is None:
            pass
        else:
            if df is None:
                pass
            else:
                raise TypeError('df must be a pd.DataFrame')

    return p


def scatter(x, y, df=None, figsize=(16, 8), marker='o', s=25, color='black', edgecolor='none', alpha=0.9, ticklabelsize=10):

    x, y = check_df(x, y, df)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, marker=marker, s=s, color=color, edgecolor=edgecolor, alpha=alpha)

    plot_style(ax, is_bar=False)

    ax = range_frame(ax, x, y, fontsize=ticklabelsize)

    return fig, ax


def line(x, y, df=None, figsize=(16, 8), linestyle='tufte', linewidth=1.0, color='black', alpha=0.9, ticklabelsize=10, markersize=10, **kwargs):

    """
    This is the line plot function, which defaults to Tufte-style with circle markers.
    If something other than 'tufte' is selected, let user set marker defaults.
    """

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

    # Is label provided?
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
