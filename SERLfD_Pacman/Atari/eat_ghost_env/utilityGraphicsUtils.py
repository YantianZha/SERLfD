# utilityGraphicsUtils.py
# ----------------

import sys
import time
import tkinter
import os.path
from pacman_src import graphicsUtils

_Windows = sys.platform == 'win32'  # True if on Win95/98/NT

_utility_window = None      # The root window for graphics output
_utility_canvas = None      # The canvas which holds graphics
_utility_canvas_xs = None      # Size of canvas object
_utility_canvas_ys = None
_utility_canvas_x = None      # Current position on canvas
_utility_canvas_y = None
_utility_canvas_col = None      # Current colour (set to black below)
_utility_canvas_tsize = 12
_utility_canvas_tserifs = 0


# define function from graphicsUtils
formatColor = graphicsUtils.formatColor
colorToVector = graphicsUtils.colorToVector

if _Windows:
    _utility_canvas_tfonts = ['times new roman', 'lucida console']
else:
    _utility_canvas_tfonts = ['times', 'lucidasans-24']
    pass  # XXX need defaults here


def sleep(secs):
    global _utility_window
    if _utility_window == None:
        time.sleep(secs)
    else:
        _utility_window.update_idletasks()
        _utility_window.after(int(1000 * secs), _utility_window.quit)
        _utility_window.mainloop()


def begin_graphics(width=640, height=480, color=formatColor(0, 0, 0), title=None):
    global _utility_window, _utility_canvas, _utility_canvas_x, _utility_canvas_y, _utility_canvas_xs, _utility_canvas_ys, _bg_color

    # Check for duplicate call
    if _utility_window is not None:
        # Lose the window.
        _utility_window.destroy()

    # Save the canvas size parameters
    _utility_canvas_xs, _utility_canvas_ys = width - 1, height - 1
    _utility_canvas_x, _utility_canvas_y = 0, _utility_canvas_ys
    _bg_color = color

    # Create the root window
    _utility_window = tkinter.Tk()
    _utility_window.protocol('WM_DELETE_WINDOW', _destroy_window)
    _utility_window.title(title or 'Graphics Window')
    _utility_window.resizable(0, 0)

    # Place the utility window below the main window
    _utility_window.geometry('%dx%d+%d+%d' % (width, height, 10, int(height*1.5)))

    # Create the canvas object
    try:
        _utility_canvas = tkinter.Canvas(_utility_window, width=width, height=height)
        _utility_canvas.pack()
        draw_background()
        _utility_canvas.update()
    except:
        _utility_window = None
        raise


def draw_background():
    corners = [(0, 0), (0, _utility_canvas_ys),
               (_utility_canvas_xs, _utility_canvas_ys), (_utility_canvas_xs, 0)]
    polygon(corners, _bg_color, fillColor=_bg_color,
            filled=True, smoothed=False)


def _destroy_window(event=None):
    sys.exit(0)


def end_graphics():
    global _utility_window, _utility_canvas, _mouse_enabled
    try:
        try:
            sleep(1)
            if _utility_window != None:
                _utility_window.destroy()
        except SystemExit as e:
            print(('Ending graphics raised an exception:', e))
    finally:
        _utility_window = None
        _utility_canvas = None
        _mouse_enabled = 0


def clear_screen(background=None):
    global _utility_canvas_x, _utility_canvas_y
    _utility_canvas.delete('all')
    draw_background()
    _utility_canvas_x, _utility_canvas_y = 0, _utility_canvas_ys


def polygon(coords, outlineColor, fillColor=None, filled=1, smoothed=1, behind=0, width=1):
    c = []
    for coord in coords:
        c.append(coord[0])
        c.append(coord[1])
    if fillColor == None:
        fillColor = outlineColor
    if filled == 0:
        fillColor = ""
    poly = _utility_canvas.create_polygon(
        c, outline=outlineColor, fill=fillColor, smooth=smoothed, width=width)
    if behind > 0:
        _utility_canvas.tag_lower(poly, behind)  # Higher should be more visible
    return poly


def square(pos, r, color, filled=1, behind=0):
    x, y = pos
    coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    return polygon(coords, color, color, filled, 0, behind=behind)


def circle(pos, r, outlineColor, fillColor, endpoints=None, style='pieslice', width=2):
    x, y = pos
    x0, x1 = x - r - 1, x + r
    y0, y1 = y - r - 1, y + r
    if endpoints == None:
        e = [0, 359]
    else:
        e = list(endpoints)
    while e[0] > e[1]:
        e[1] = e[1] + 360

    return _utility_canvas.create_arc(x0, y0, x1, y1, outline=outlineColor, fill=fillColor,
                              extent=e[1] - e[0], start=e[0], style=style, width=width)


def image(pos, file="../../blueghost.gif"):
    x, y = pos
    # img = PhotoImage(file=file)
    return _utility_canvas.create_image(x, y, image=tkinter.PhotoImage(file=file), anchor=tkinter.NW)


def refresh():
    _utility_canvas.update_idletasks()


def moveCircle(id, pos, r, endpoints=None):
    global _utility_canvas_x, _utility_canvas_y

    x, y = pos
#    x0, x1 = x - r, x + r + 1
#    y0, y1 = y - r, y + r + 1
    x0, x1 = x - r - 1, x + r
    y0, y1 = y - r - 1, y + r
    if endpoints == None:
        e = [0, 359]
    else:
        e = list(endpoints)
    while e[0] > e[1]:
        e[1] = e[1] + 360

    if os.path.isfile('flag'):
        edit(id, ('extent', e[1] - e[0]))
    else:
        edit(id, ('start', e[0]), ('extent', e[1] - e[0]))
    move_to(id, x0, y0)


def edit(id, *args):
    _utility_canvas.itemconfigure(id, **dict(args))


def text(pos, color, contents, font='Helvetica', size=12, style='normal', anchor="nw"):
    global _utility_canvas_x, _utility_canvas_y
    x, y = pos
    font = (font, str(size), style)
    return _utility_canvas.create_text(x, y, fill=color, text=contents, font=font, anchor=anchor)


def changeText(id, newText, font=None, size=12, style='normal'):
    _utility_canvas.itemconfigure(id, text=newText)
    if font != None:
        _utility_canvas.itemconfigure(id, font=(font, '-%d' % size, style))


def changeColor(id, newColor):
    _utility_canvas.itemconfigure(id, fill=newColor)


def line(here, there, color=formatColor(0, 0, 0), width=2):
    x0, y0 = here[0], here[1]
    x1, y1 = there[0], there[1]
    return _utility_canvas.create_line(x0, y0, x1, y1, fill=color, width=width)


def remove_from_screen(x,
                       d_o_e=lambda arg: _utility_window.dooneevent(arg),
                       d_w=tkinter._tkinter.DONT_WAIT):
    _utility_canvas.delete(x)
    d_o_e(d_w)


def _adjust_coords(coord_list, x, y):
    for i in range(0, len(coord_list), 2):
        coord_list[i] = coord_list[i] + x
        coord_list[i + 1] = coord_list[i + 1] + y
    return coord_list


def move_to(object, x, y=None,
            d_o_e=lambda arg: _utility_window.dooneevent(arg),
            d_w=tkinter._tkinter.DONT_WAIT):
    if y is None:
        try:
            x, y = x
        except:
            raise Exception('incomprehensible coordinates')

    horiz = True
    newCoords = []
    current_x, current_y = _utility_canvas.coords(object)[0:2]  # first point
    for coord in _utility_canvas.coords(object):
        if horiz:
            inc = x - current_x
        else:
            inc = y - current_y
        horiz = not horiz

        newCoords.append(coord + inc)

    _utility_canvas.coords(object, *newCoords)
    d_o_e(d_w)


def move_by(object, x, y=None,
            d_o_e=lambda arg: _utility_window.dooneevent(arg),
            d_w=tkinter._tkinter.DONT_WAIT, lift=False):
    if y is None:
        try:
            x, y = x
        except:
            raise Exception('incomprehensible coordinates')

    horiz = True
    newCoords = []
    for coord in _utility_canvas.coords(object):
        if horiz:
            inc = x
        else:
            inc = y
        horiz = not horiz

        newCoords.append(coord + inc)

    _utility_canvas.coords(object, *newCoords)
    d_o_e(d_w)
    if lift:
        _utility_canvas.tag_raise(object)


def writePostscript(filename):
    """ Writes the current canvas to a postscript file. """
    psfile = open(filename, 'w')
    psfile.write(_utility_canvas.postscript(pageanchor='sw', y='0.c', x='0.c'))
    psfile.close()


def getPostscript():
    """ return the postscript of current utility canvas """
    return _utility_canvas.postscript(pageanchor='sw', y='0.c', x='0.c', pagewidth=_utility_canvas_xs, pageheight=_utility_canvas_xs)


ghost_shape = [
    (0, - 0.5),
    (0.25, - 0.75),
    (0.5, - 0.5),
    (0.75, - 0.75),
    (0.75, 0.5),
    (0.5, 0.75),
    (- 0.5, 0.75),
    (- 0.75, 0.5),
    (- 0.75, - 0.75),
    (- 0.5, - 0.5),
    (- 0.25, - 0.75)
]

if __name__ == '__main__':
    begin_graphics()
    clear_screen()
    ghost_shape = [(x * 10 + 20, y * 10 + 20) for x, y in ghost_shape]
    g = polygon(ghost_shape, formatColor(1, 1, 1))
    move_to(g, (50, 50))
    circle((150, 150), 20, formatColor(0.7, 0.3, 0.0), endpoints=[15, - 15], fillColor=formatColor(255.0/255.0, 255.0/255.0, 61.0/255))
    sleep(2)
