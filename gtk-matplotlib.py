import sys
import logging

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar
from matplotlib.patches import Rectangle

class MainClass(Gtk.Window):
    def __init__(self, title='Matplotlib', size=(800, 500)):
        Gtk.Window.__init__(self, title=title)
        self.set_default_size(*size)
        self.boxvertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.connect("delete-event", Gtk.main_quit)
        self.add(self.boxvertical)

        # self.toolbar = Gtk.Toolbar()
        # self.context = self.toolbar.get_style_context()
        # self.context.add_class(Gtk.STYLE_CLASS_PRIMARY_TOOLBAR)
        # self.boxvertical.pack_start(self.toolbar, False, False, 0)
        #
        # self.addbutton = Gtk.ToolButton(Gtk.STOCK_ADD)
        # self.removebutton = Gtk.ToolButton(Gtk.STOCK_REMOVE)
        #
        # self.toolbar.insert(self.addbutton, 0)
        # self.toolbar.insert(self.removebutton, 1)

        self.box = Gtk.Box()
        self.boxvertical.pack_start(self.box, True, True, 0)


        # This can be put into a figure or a class ####################
        self.fig = plt.Figure(figsize=(10, 10), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        ###############################################################
        self.box.pack_start(self.canvas, True, True, 0)

        # self.addbutton.connect("clicked", self.addrow)
        # self.removebutton.connect("clicked", self.removerow)

        self.box_buttons = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.box.pack_start(self.box_buttons, True, True, 0)
        # self.liststore = Gtk.ListStore(float, float)
        # self.treeview = Gtk.TreeView(model=self.liststore)
        # self.box_buttons.pack_start(self.treeview, False, True, 0)

        self.buttons_toolbar = Gtk.Toolbar()
        self.context2 = self.buttons_toolbar.get_style_context()
        self.context2.add_class(Gtk.STYLE_CLASS_PRIMARY_TOOLBAR)
        self.box_buttons.pack_start(self.buttons_toolbar, False, False, 0)

        self.refreshbutton = Gtk.ToolButton(Gtk.STOCK_REFRESH)
        self.buttons_toolbar.insert(self.refreshbutton, 0)

        self.refreshbutton.connect("clicked", self.refresh)

        self.toolbar2 = NavigationToolbar(self.canvas, self)
        self.boxvertical.pack_start(self.toolbar2, False, True, 0)

        self.statbar = Gtk.Statusbar()
        self.boxvertical.pack_start(self.statbar, False, True, 0)

        self.fig.canvas.mpl_connect('motion_notify_event', self.updatecursorposition)

        # self.xrenderer = Gtk.CellRendererText()
        # self.xrenderer.set_property("editable", True)
        # self.xcolumn = Gtk.TreeViewColumn("x-Value", self.xrenderer, text=0)
        # self.xcolumn.set_min_width(100)
        # self.xcolumn.set_alignment(0.5)
        # self.treeview.append_column(self.xcolumn)

        # self.yrenderer = Gtk.CellRendererText()
        # self.yrenderer.set_property("editable", True)
        # self.ycolumn = Gtk.TreeViewColumn("y-Value", self.yrenderer, text=1)
        # self.ycolumn.set_min_width(100)
        # self.ycolumn.set_alignment(0.5)
        # self.treeview.append_column(self.ycolumn)

        # self.xrenderer.connect("edited", self.xedited)
        # self.yrenderer.connect("edited", self.yedited)

        # self.liststore.append([2.35, 2.40])
        # self.liststore.append([3.45, 4.70])
        self.xdata = np.linspace(0, 3*np.pi, 10000)
        self.freq = 1.0
        self.ydata = np.sin(self.xdata * 2.0 * np.pi * self.freq)
        self.p1, = self.ax.plot(self.xdata, self.ydata)
        self.fig.canvas.draw()

    def resetplot(self):
        # self.ax.cla()
        # self.ax.set_xlim(0, 10)
        # self.ax.set_ylim(0, 10)
        self.ax.grid(True)

    def refresh(self, event):
        self.freq += 1
        self.ydata = np.sin(self.xdata * 2.0 * np.pi * self.freq)
        self.plotpoints()


    def plotpoints(self):
        self.resetplot()
        self.p1.set_xdata(self.xdata)
        self.p1.set_ydata(self.ydata)
        self.fig.canvas.draw()

    def xedited(self, widget, path, number):
        self.liststore[path][0] = float(number.replace(',', '.'))
        self.plotpoints()

    def yedited(self, widget, path, number):
        self.liststore[path][1] = float(number.replace(',', '.'))
        self.plotpoints()

    def addrow(self, widget):
        self.liststore.append()
        self.plotpoints()

    def removerow(self, widget):
        self.select = self.treeview.get_selection()
        self.model, self.treeiter = self.select.get_selected()
        if self.treeiter is not None:
            self.liststore.remove(self.treeiter)
        self.plotpoints()

    def updatecursorposition(self, event):
        '''When cursor inside plot, get position and print to statusbar'''
        if event.inaxes:
            x = event.xdata
            y = event.ydata
            self.statbar.push(1, ("Coordinates:" + " x= " + str(round(x, 3)) + "  y= " + str(round(y, 3))))


mc = MainClass(title='Scatter creator', size=(1000, 700))

mc.resetplot()
mc.plotpoints()

mc.show_all()
Gtk.main()