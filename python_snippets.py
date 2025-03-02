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

# -----------------------------------------------------------------------------

# matplotlib snippets

# Get matplotlib version
import matplotlib
print(matplotlib.__version__)
"""
3.5.1
"""

# Set Matplotlib backend with magic in Jupyter notebook.
# %matplotlib notebook # interactive plots
# %matplotlib inline # static plots
# %matplotlib nbagg # output only
# https://matplotlib.org/stable/users/explain/figure/backends.html
# https://stackoverflow.com/questions/4930524/how-can-i-set-the-matplotlib-backend

# Basic plot in Jupyter notebook
fig, ax = plt.subplots(constrained_layout=True)
x = np.linspace(0, 10)
y = np.sin(x)
ax.plot(x, y, '.-');

fig.canvas.draw();

plt.close(fig); del fig, ax;

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
    facecolor="w",
);

# Change font size of axis label.
fig.set_xlabel("this is the x-axis", fontsize=20)
fig.set_ylabel("this is the y-axis", fontsize=20)

# Instantiate a second axes that shares the same x-axis
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

# Turn off frame / border around legend
ax.legend(frameon=False)
# https://stackoverflow.com/questions/25540259/remove-or-adapt-border-of-frame-of-legend-using-matplotlib
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

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
