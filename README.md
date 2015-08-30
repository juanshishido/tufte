# Tufte in Python

See the [Jupyter notebook](https://github.com/juanshishido/tufte/blob/master/tufte-in-python.ipynb) for more details.

A set of four plots based on Edward R. Tufte's designs in _The Visual Display of Quantitative Information_:

* bar
* boxplot
* line
* scatter

All of these plots aim to maximize _data-ink_, the "non-erasable core of a graphic." 

![bar](/images/bar.png)

The "boxplot" (`bplot`), for example, removes boxes and caps and simply shows a dot between two lines. The dot represents the median and the lines correspond to the top and bottom 25% of the data. The empty space between the lines is the interquartile range.

![bplot](/images/bplot.png)

The line and scatter plots make use of Tufte's _range-frame_ concept, which aims to make the frame (axis) lines "effective data-communicating element[s]" by showing the minimum and maximum values in each axis. The default line style uses a circle marker with gaps between line segments.

![line](/images/line.png)

![scatter](/images/scatter.png)

This is built on top of `matplotlib`. This means other functions or methods can be used in conjunction with `tufte` plots.

Note: plots shown for demonstration purposes only, thus no titles or axis labels are used.
