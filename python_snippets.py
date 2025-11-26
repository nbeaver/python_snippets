#! /usr/bin/env python3

import sys
sys.exit(0)
# Put this at the top so executing this does nothing.

# -----------------------------------------------------------------------------

# Printing and formatting

# Print without a trailing newline (Python3)
print("no newline", end='')
for i in range(10):
    print("{i} ".format(i), end='')

# Formatting
# https://docs.python.org/3/library/string.html#format-specification-mini-language

# Exponentials
print("%e" % 2**32)
"""
4.294967e+09
"""
print('{:e}'.format(2**32))
"""
4.294967e+09
"""

# Percentages
print("{:2.2%}".format(0.34567))
"""
34.57%
"""

# Zero padding / print integer with leading zeros
print('{:03d}'.format(42))
"""
042
"""
print("{:02d}".format(1))
"""
01
"""

# Print current timestamp with timezone
ctime_with_tz = "%a %b %d %H:%M:%S %Y %Z"
long_time = "%A, %B %d, %Y, %I:%M:%S %p %Z"
now = datetime.datetime.now().astimezone()
print("local time: {}".format(now.strftime(ctime_with_tz)))
print("local time: {}".format(now.strftime(long_time)))
print("iso format: {}".format(now.isoformat()))
"""
local time: Fri Apr 05 10:46:26 2024 Eastern Daylight Time
local time: Friday, April 05, 2024, 10:46:26 AM Eastern Daylight Time
iso format: 2024-04-05T10:46:26.160316-04:00
"""
# https://stackoverflow.com/questions/311627/how-to-print-a-date-in-a-regular-format
# https://stackoverflow.com/questions/31299580/python-print-the-time-zone-from-strftime
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
# https://strftime.org/

# -----------------------------------------------------------------------------

# Get first key from a dictionary
next(iter(mydict))

# Arbitrary container class
class MyInfo:
    # Just make this a container for attributes.
    pass

class MyInfo:
    # Give names of class members.
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'
    def __str__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'

# https://stackoverflow.com/questions/32994545/standard-python-base-class-as-a-container-for-arbitrary-attributes
# https://stackoverflow.com/questions/17595296/why-do-pythons-empty-classes-and-functions-work-as-arbitrary-data-containers-b

# Iterate over dictionary in value order
for key, val in sorted(dictionary.items(), key=lambda x: x[1]):
    print(key, val)

# https://stackoverflow.com/questions/674509/how-do-i-iterate-over-a-python-dictionary-ordered-by-values

# Iterable unpacking in function calls.
mylist = [1, 2, 3]
mydict = {1: 'a', 2: 'b', 3: 'c'}
print(*mylist, **mydict)
# Note: it's hard to get help on iterable unpacking / starred expressions with '*'
# since help('*') is for the multiplication operator.
# https://docs.python.org/3/reference/expressions.html
# https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
# https://stackoverflow.com/questions/12555627/python-3-starred-expression-to-unpack-a-list

# Modify a list in-place.
mylist[:] = [1, 2, 3]
# https://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk#19859907

# Check if a script has been run with more than one argument, and print usage otherwise.
if len(sys.argv) > 1:
    rootdir = sys.argv[1]
else:
    sys.stderr.write("Usage: {} --args-go-here\n".format(sys.argv[0]))
    sys.exit(1)

# -----------------------------------------------------------------------------

# Interactively look at variables.

# Get global variables.
globals()

# Get local variables.
locals()

# See functions in 'math' module.
import math; vars(math)

# Find out about the methods of an object, e.g. the integer 2.
dir(2)
"""
['__abs__', '__add__', '__and__', '__bool__',....
"""

n = 2**12-1
print(n, n.bit_length())
"""
4095 12
"""
n = 2**12
print(n, n.bit_length())
"""
4096 13
"""

# Get help on a method
help(object.method)

# Get help on a package
help(numpy)

# Get help on opening files.
help(open)

# print docstring
print(open.__doc__)

# To drop into an interactive shell from a script:
import code
code.interact(local=locals())

# Using ipython/ipydb to get a better interactive shell.
import ipdb
# At point of interest:
ipdb.set_trace()

# -----------------------------------------------------------------------------

# random samples

# random phone number
rand = random.randint(10**9,int('9' * 10))
"""
5746885699
"""
# Put the hyphens in
s = str(rand)
s[:3] +"-"+ s[3:6] +"-"+ s[6:]
"""
'574-688-5699'
"""

# Use reduce() function with a lambda expression.
reduce(lambda x, y: x*y, [1, 2, 3, 4, 5])

# Check if a module is run standalone:
if __name__ == '__main__':
    print("Running standalone.")


# Check where local modules should be installed.

import site
print(site.USER_SITE)
"""
'/home/nathaniel/.local/lib/python3.10/site-packages'
"""
# $ python -m site --user-site

# Note this does not adapt for e.g. anaconda environments, for that use sys.path:
import sys
print(sys.path)

# -----------------------------------------------------------------------------

# Numpy
# Slicing along different axes
arr2d = np.array([[3, 6, 9, 12, 15], [4, 8, 12, 16, 20]])
"""
array([[ 3,  6,  9, 12, 15],
       [ 4,  8, 12, 16, 20]])
"""
arr2d[1] # slicing along axis 0
"""
array([ 4,  8, 12, 16, 20])
"""
arr2d[:, 2]# slicing along axis 1
"""
array([ 6, 8])
"""

# Unwrapping a singleton array
numpy.item(myarray)
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
# https://stackoverflow.com/questions/35157742/how-to-convert-singleton-array-to-a-scalar-value-in-python
# https://stackoverflow.com/questions/34822442/how-to-convert-neatly-1-size-numpy-array-to-a-scalar-numpy-asscalar-gives-err

# Reducing shape of an array with singleton elements
np.zeros((2, 1, 1, 3)).shape
"""
(2,1,1,3)
"""
np.squeeze(np.zeros((2, 1, 1, 3))).shape
"""
(2,3)
"""

# Discarding rows with NaNs
arr2d_no_NaNs = arr2d[~np.isnan(arr2d).any(axis=1)]

# Formatting in numpy
np.set_printoptions(threshold=60) # truncate if array is longer than 60 elements
np.set_printoptions() # reset

# Loading a CSV file, skipping the first 3 rows.
columns = np.loadtxt(filepath, skiprows=3, unpack=True, delimiter=',')

# -----------------------------------------------------------------------------

# matplotlib snippets

# Get matplotlib version
import matplotlib
print(matplotlib.__version__)
"""
3.5.1
"""

import matplotlib.pyplot as plt

# Set Matplotlib backend with magic in Jupyter notebook.
# %matplotlib notebook # interactive plots
# %matplotlib inline # static plots
# %matplotlib nbagg # output only
# https://matplotlib.org/stable/users/explain/figure/backends.html
# https://stackoverflow.com/questions/4930524/how-can-i-set-the-matplotlib-backend

# Basic plot in Jupyter notebook
import matplotlib.pyplot as plt
fig, ax = plt.subplots(constrained_layout=True)
x = np.linspace(0, 10)
y = np.sin(x)
ax.plot(x, y, '.-');

fig.canvas.draw();

plt.close(fig); del fig, ax;

# Plot with error bars.
ax.errorbar(x, y, yerr=y_err, fmt='.', capsize=2, label="error bars")
# Customize error bar thickness.
ax.errorbar(x, y, yerr=y_err, fmt='.', capsize=2, markersize=2, linewidth=0.6, elinewidth=0.8, linestyle='')
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html

# Scatter plot, color based on z.
paths = ax.scatter(x, y, c=z, cmap='plasma', s=3)
fig.colorbar(paths, ax=ax, label='z [units]')
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

# Custom colorbar for series of line plots.
zs = np.linspace(0, 10, num=20)
sm_all = plt.cm.ScalarMappable(cmap='copper', norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
x = np.linspace(1, 5, num=50)
ys = [x.copy()*i*0.5 + i/5 for i in range(20)]
fig, ax = plt.subplots(constrained_layout=True)
for i, (y,z) in enumerate(zip(ys,zs)):
    ax.plot(x, y, '.-', color=sm_all.to_rgba(z))
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(sm_all, ax=ax, label='z')
# https://stackoverflow.com/questions/8342549/add-colorbar-to-a-sequence-of-line-plots
# https://stackoverflow.com/questions/26545897/drawing-a-colorbar-aside-a-line-plot-using-matplotlib
# https://stackoverflow.com/questions/30779712/show-matplotlib-colorbar-instead-of-legend-for-multiple-plots-with-gradually-cha?noredirect=1&lq=1

# Make a histogram
bin_vals, bin_edges, patches = plt.hist(numbers, bins=n_bins)
ax.hist(numbers, bins='auto', alpha=0.8, histtype='stepfilled')
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html

ax.axvline(x=stdevs.mean(),label="mean of stdev", linestyle="--",color="blue")

# Save to file
fig.savefig(
    "example.png",
    bbox_inches='tight',
    metadata = {"Title": "example title", "Author": "Firstname Lastname"},
    dpi=200,
    facecolor="w", # white background
);
# For PDF, metadata fields be in:
# 'Title', 'Producer', 'Trapped', 'ModDate', 'Subject', 'Keywords', 'CreationDate', 'Creator', 'Author'

# Example notebook ID
nb_id = 1706739890
import time
try:
    print(nb_id)
except NameError:
    print(round(time.time()))

# Change font size of axis label.
fig.set_xlabel("this is the x-axis", fontsize=20)
fig.set_ylabel("this is the y-axis", fontsize=20)

# Instantiate a second axes that shares the same x-axis
# Both axes are plotted on top of each other.
fig, ax1 = plt.subplots(constrained_layout=True)
ax2 = ax1.twinx()
ax1.set_ylabel("first y-label")
ax2.set_ylabel("second y-label")

fig.supxlabel("x-label for both axes")
fig.suptitle("title for both axes")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
# https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend/10129461#10129461

# Extra for twinx axes:
ax1.plot([], [], color='tab:orange') # dummy plot
ax1.legend() # show the legend for both
# Color axis #2 differently
ax2.spines['right'].set_color('tab:orange')
ax2.xaxis.label.set_color('tab:orange')
ax2.tick_params(axis='y', colors='tab:orange')

# Plot like this:
#   ----------------
# y1|              |
#   |              |
#   ----------------
# y2|              |
#   ----------------
#          x
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [2, 1]}
)
fig.set_constrained_layout_pads(h_pad=0, hspace=0, w_pad=0, wspace=0.0)
ax1.plot(x1, y1)
ax1.set_ylabel("y1")
ax2.plot(x2, y1)
ax2.set_ylabel("y2")
fig.supxlabel("x (both)");

# Turn off frame / border around legend
ax.legend(frameon=False)
# https://stackoverflow.com/questions/25540259/remove-or-adapt-border-of-frame-of-legend-using-matplotlib
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

# Make legend opaque (frame must be on).
ax.legend(framealpha=1.0, frameon=True)
# Default framealpha = 0.8.
# https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity

# Change color and size of text (default 10).
ax.legend(labelcolor='green', fontsize=12)

# Turn off frame / border / bounding box.
ax.set_frame_on(False)

# Turn off axis labels, tick marks, etc.
ax.axis('off')

# See current figure size and DPI settings.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']
"""
[6.4, 4.8]
"""
plt.rcParams['figure.dpi']
"""
100.0
"""
# https://stackoverflow.com/questions/56231689/how-to-set-the-default-figure-size-and-dpi-of-all-plots-drawn-by-matplotlib-pyp

# Set default markersize.

rcParams['lines.markersize']
"""
6.0
"""

# Linestyles
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# named linestyles
# 'solid', '-'
# 'dotted', ':'
# 'dashed', '--'
# 'dashdot', '-.'

# Heatmaps with non-uniform meshes
plot = ax.pcolormesh(
    H_order_calibrated,
    angle_order,
    S21_logmag_stack,
    cmap='magma',
)
fig.colorbar(mappable=plot, ax=ax, label='A.U.')
# https://stackoverflow.com/questions/72035916/how-to-use-matplotlibs-pcolormesh-with-non-uniform-mesh
# https://stackoverflow.com/questions/19572409/matplotlib-heatmap-with-changing-y-values

# Heatmap with logarithmic scaling.
plot = ax.pcolormesh(
    H_order_calibrated,
    angle_order,
    S21_logmag_stack,
    cmap='magma',
    norm='log'
)
# https://stackoverflow.com/questions/17201172/a-logarithmic-colorbar-in-matplotlib-scatter-plot
# https://matplotlib.org/stable/users/explain/colors/colormapnorms.html#logarithmic


# Adjust colorbar height and and how close it is to the heatmap.
fig.colorbar(mappable=plot, ax=ax, fraction=0.046, pad=0.01, label='z [units]')
# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

# Add label to colorbar and change font size.
cbar = fig.colorbar(mappable=plot, ax=ax, label='z [units]')
cbar.ax.tick_params(labelsize=18) # font size of nmber on colorbar
cbar.set_label(label='a label',size=15,weight='bold')
# https://stackoverflow.com/questions/23172282/how-to-change-font-properties-of-a-matplotlib-colorbar-label

# Invert y-axis so larger values are on the bottom
axs.invert_yaxis()
# OR
ax.yaxis.set_inverted(True)
# Note: must come after ax.set_ylim()
# https://stackoverflow.com/questions/2051744/how-to-invert-the-x-or-y-axis

# Plot datetime on the x-axis
fig, ax = plt.subplots()
ax.plot(time_parsed, y_vals, '.-')
ax.set_xlabel("timestamp")
ax.set_ylabel('y_vals');
xfmt = matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(xfmt)
fig.autofmt_xdate();

# Plot a rectangle
import matplotlib.pyplot as plt
ax.add_patch(
    plt.Rectangle(
        (0,0), # bottom left corner (for positive width)
        3.0, # width
        4.0, # height
        lw=1,
        facecolor='none',
        edgecolor='black'),
        fill = False,
)
"""
  +------------------+
  |                  |
height               |
  |                  |
 (xy)---- width -----+
"""
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

# Plot an arrow
ax.arrow(
    10.5, 5, # x,y
    -2, 0.0, # dx, dy
    head_width=0.1,
    shape='full',
    color='red',
)

# Plot with a square viewport.
ax.set_aspect('equal')
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_aspect.html

# Click on a point to get coordinates (in Jupyter)
def on_click(event):
    global poi_x
    global poi_y
    global ax
    poi_x = event.xdata
    poi_y = event.ydata
    ax.scatter(event.xdata, event.ydata, color="green")
if choose_poi:
    fig.canvas.mpl_connect('button_press_event', on_click)

# Plot text in data coordinates.
ax.text(3.4, 52, r'this is text, $\alpha^2$')

# Plot text in axis coordinates.
ax.text(0.5, 0.5, r'this is text, $\alpha^2$',
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes,
    fontsize=16,
)

# Plot text with data coordinates in x but axis coordinates in y.
ax.text(0.5, 0.5, r'this is text, $\alpha^2$',
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes,
    fontsize=16,
)
# https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html
# https://stackoverflow.com/questions/63153629/use-data-coords-for-x-axis-coords-for-y-for-text-annotations

# Add annotate (with arrow) but no text.
ax.annotate(
    "",
    xy=(-35, -20),
    xytext=(-35,-20),
    arrowprops=dict(
        width=4,
        headwidth=15,
        headlength=15,
        color='black',
#         arrowstyle="-|>,head_width=0.4,head_length=0.8",
#         shrinkA=0,
#         shrinkB=0
    ),
    color='black',
    fontsize=18,
    xycoords='data',
    textcoords='data',
#     transform = ax.transAxes,
)
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html

# Skip part of an axis range, i.e. a gap or broken axis.
# This is for an x-axis.
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x1, y1)
ax2.plot(x2, y2)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax2.tick_params(labelright='on')
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)
fig.subplots_adjust(wspace=.08)
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
# https://stackoverflow.com/questions/5656798/is-there-a-way-to-make-a-discontinuous-axis-in-matplotlib

# Add scalebar.
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
fontprops = matplotlib.font_manager.FontProperties(size=18)
scalebar = AnchoredSizeBar(
    ax.transData,
    20,
    '20 μm',
    'lower left',
    pad=2.2,
    color='white',
    frameon=False,
    size_vertical=1,
    fontproperties=fontprops,
)
ax.add_artist(scalebar)
# https://stackoverflow.com/questions/39786714/how-to-insert-scale-bar-in-a-map-in-matplotlib

# Hide axes
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Hide ticks and tick labels.
ax.set_xticklabels([]);
ax.set_xticks([])
# https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots

# Hide everything but the plotted lines.
ax.axis('off');
# https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-a-figure

# Interactive clicking on plots in Jupyter notebooks (mouse events).
def on_click(event):
    global poi_n_x
    global poi_n_y
    global ax
    poi_n_x = event.xdata
    poi_n_y = event.ydata
    ax.scatter(event.xdata, event.ydata, color="green")

def on_click(event):
    global ax
    global OLD_AXVLINE
    global freq
    global near_index
    global chosen_freqs
    raw_value = event.xdata
    near_index = closest_index(freq*GHz, raw_value)
    chosen_freq = freq[near_index]
    if len(chosen_freqs) < n_lorentzians:
        chosen_freqs.append(chosen_freq)
        ax.axvline(chosen_freq*GHz, color="black", linestyle='--')

fig.canvas.mpl_connect('button_press_event', on_click)
# https://matplotlib.org/stable/users/explain/figure/event_handling.html
# https://stackoverflow.com/questions/15032638/how-to-return-a-value-from-button-press-event-matplotlib

# Throw an assertion error if there are still active plots left over.
assert plt.get_fignums() == []

mat_dict = scip.io.loadmat("example.mat", simplify_cells=True)
