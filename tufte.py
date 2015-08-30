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


def plot_style(ax, plot_type):
    ax.tick_params(axis='both', top='off', bottom='off', left='off', right='off', colors='#4B4B4B', pad=10)
    ax.xaxis.label.set_color('#4B4B4B')
    ax.yaxis.label.set_color('#4B4B4B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plot_type.lower() == 'bar':
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.75)
        ax.spines['bottom'].set_edgecolor('LightGray')
    elif plot_type.lower() == 'bplot':
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', left='on')
    elif plot_type.lower() in ('line', 'scatter'):
        ax.spines['left'].set_linewidth(0.75)
        ax.spines['bottom'].set_linewidth(0.75)
        ax.spines['left'].set_edgecolor('#4B4B4B')
        ax.spines['bottom'].set_edgecolor('#4B4B4B')

def all_ints(data):
    if isinstance(data, pd.DataFrame):
        d_temp = []
        for c in data.columns:
            d_temp = d_temp + data[c].tolist()
        data = d_temp
    if type(data) not in (list, np.ndarray, pd.Series):
        raise TypeError('Container must be of type: list, np.ndarray, or pd.Series')
    return sum([float(v).is_integer() for v in data]) == len(data)

def cast_to(kind=float, labels=None):
    if kind == 'float':
        labels = [round(float(v), 1) for v in labels]
    elif kind == 'int':
        labels = [int(v) for v in labels]
    else:
        raise TypeError('kind must be either float or int')
    return labels

def convert_ticks(data, labels):
    if all_ints(data):
        labels = cast_to('int', labels)
    else:
        labels = cast_to('float', labels)
    return labels

def range_frame(fontsize, ax, x=None, y=None, dimension='both', is_bar=False):
    PAD = 0.05
    if dimension in ('x', 'both'):
        assert x is not None, 'Must pass in x value'
        xmin = x.min().min()
        xmax = x.max().max()
        xlower = xmin - ((xmax - xmin) * PAD)
        xupper = xmax + ((xmax - xmin) * PAD)
        ax.set_xlim(xmin=xlower, xmax=xupper)
        ax.spines['bottom'].set_bounds(xmin, xmax)
        xlabels = [xl for xl in ax.xaxis.get_majorticklocs() if xl > xmin and xl < xmax]
        xlabels = [xmin] + xlabels + [xmax]
        xlabels = convert_ticks(x, xlabels)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=fontsize)
    if dimension in ('y', 'both'):
        assert y is not None, 'Must pass in y value'
        ymin = y.min().min()
        ymax = y.max().max()
        ylower = ymin - ((ymax - ymin) * PAD)
        yupper = ymax + ((ymax - ymin) * PAD)
        if is_bar:
            ax.set_ylim(ymin=0, ymax=yupper) 
            ax.spines['left'].set_bounds(0, ymax)
            ylabels = [yl for yl in ax.yaxis.get_majorticklocs() if yl < ymax]
            ylabels = ylabels + [ymax]
        else:
            ax.set_ylim(ymin=ylower, ymax=yupper) 
            ax.spines['left'].set_bounds(ymin, ymax)
            ylabels = [yl for yl in ax.yaxis.get_majorticklocs() if yl > ymin and yl < ymax]
            ylabels = [ymin] + ylabels + [ymax]
        ylabels = convert_ticks(y, ylabels)
        ax.set_yticks(ylabels)
        ax.set_yticklabels(ylabels, fontsize=fontsize)
    return ax

def auto_rotate_xticklabel(fig, ax):
    figw = fig.get_figwidth()
    nticks = len(ax.xaxis.get_majorticklocs())
    tick_spacing = (figw / float(nticks))
    font_size = [v.get_fontsize() for v in ax.xaxis.get_majorticklabels()][0]
    FONT_RATE = 0.01
    char_width = font_size * FONT_RATE
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
        raise TypeError('Container must be of type: list, np.ndarray, pd.core.index.Int64Index, or pd.Series')
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

def check_valid(data):
    if isinstance(data, pd.DataFrame):
        return True
    elif type(data) in (list, np.ndarray, pd.Series):
        return True
    else:
        return False

def scatter(x, y, df=None, figsize=(16, 8), marker='o', s=25, color='black', edgecolor='none', alpha=0.9, ticklabelsize=10):
    x, y = check_df(x, y, df)
    fig, ax = plt.subplots(figsize=figsize)
    plot_style(ax, plot_type='scatter')
    ax.scatter(x, y, marker=marker, s=s, color=color, edgecolor=edgecolor, alpha=alpha)
    ax = range_frame(ticklabelsize, ax, x, y, dimension='both')
    return fig, ax

def line(x, y, df=None, figsize=(16, 8), linestyle='tufte', linewidth=1.0, color='black', alpha=0.9, ticklabelsize=10, markersize=10, **kwargs):
    x, y = check_df(x, y, df)
    fig, ax = plt.subplots(figsize=figsize)
    plot_style(ax, plot_type='line')
    if linestyle == 'tufte':
        if len(kwargs) > 0:
            warnings.warn('Marker options are being ignored')
        marker = 'o'
        ax.plot(x, y, linestyle='-', linewidth=linewidth, color=color, alpha=alpha, zorder=1)
        ax.scatter(x, y, marker=marker, s=markersize*8, color='white', zorder=2)
        ax.scatter(x, y, marker=marker, s=markersize, color=color, zorder=3)
    else:
        ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha, markersize=markersize ** 0.5, **kwargs)
    ax = range_frame(ticklabelsize, ax, x, y, dimension='both')
    return fig, ax

def bar(position, height, df=None, label=None, figsize=(16, 8), align='center', color='LightGray', edgecolor='none', width=0.5, gridcolor='white', ticklabelsize=10):
    position, height = check_df(position, height, df)
    fig, ax = plt.subplots(figsize=figsize)
    plot_style(ax, plot_type='bar')
    ax.bar(position, height, align=align, color=color, edgecolor=edgecolor, width=width)
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
    yticklocs = ax.yaxis.get_majorticklocs()
    yticklocs = convert_ticks(height, yticklocs)
    for y in yticklocs:
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
        raise ValueError('Labels must be in: list, np.ndarray, or pd.Series')
    ax = range_frame(ticklabelsize, ax, x=None, y=height, dimension='y', is_bar=True)
    return fig, ax

def bplot(x, figsize=(16, 8), auto_figsize=True, ticklabelsize=10):
    if check_valid(x):
        fig, ax = plt.subplots(figsize=figsize)
        plot_style(ax, plot_type='bplot')
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
            ax.set_xlim(min(i_pos) - 0.5, max(i_pos) + 0.5)
            ax.set_xticks(i_pos)
            ax.set_xticklabels(x.columns)
        elif type(x) in (list, np.ndarray, pd.Series):
            x = to_nparray(x)
            v000 = x.min()
            v025 = np.percentile(x, 25)
            v050 = np.median(x)
            v075 = np.percentile(x, 75)
            v100 = x.max()
            if auto_figsize:
                fig.set_size_inches(4, 8)
            else:
                pass
            ax.plot([0, 0], [v000, v025], color='black', linewidth=0.5)
            ax.plot([0, 0], [v075, v100], color='black', linewidth=0.5)
            ax.scatter([0], [v050], color='black', s=5)
            ax.axes.get_xaxis().set_visible(False)
        xmin = x.min().min()
        xmax = x.max().max()
        x_range = xmax - xmin
        ax.set_ylim(xmin - x_range * 0.05, xmax + x_range * 0.05)
    else:
        raise TypeError('x must be type: list, np.ndarray, pd.Series, or pd.DataFrame')
    ax = range_frame(ticklabelsize, ax, x=None, y=x, dimension='y')
    return fig, ax
