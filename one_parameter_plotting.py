import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Circle, CustomJS, Slider
from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure
from ripser import Rips

from helper_functions import ripser_to_rivet_bcode, compute_landscapes


def Rips_Filtration(points, radius_range):

    rips = Rips(maxdim=1, thresh=radius_range[1], verbose=False)
    barcodes = rips.transform(points)
    H_0_Bars = barcodes[0]
    H_1_Bars = barcodes[1]

    # Plotting Parameters

    H_0_color = '#1d07ad'
    H_1_color = '#009655'
    vertical_line_color = "#FB8072"
    bar_line_width = 2
    circle_opacity = 0.2
    circle_color = "green"
    circle_line_color = "black"

    source = ColumnDataSource(data=dict(x=points[:, 0],
                                        y=points[:, 1],
                                        sizes=radius_range[0] / 2 * np.ones(points.shape[0]),
                                        )
                              )
    vline = ColumnDataSource(data=dict(s=[radius_range[0]],
                                       y=[0],
                                       angle=[np.pi / 2]
                                       )
                             )

    filt_plot = figure(title='Filtration',
                       plot_width=300,
                       plot_height=300,
                       aspect_ratio="auto",
                       height_policy="auto",
                       min_border=0,
                       toolbar_location=None,
                       match_aspect=True)
    glyph = Circle(x="x", y="y", radius="sizes", line_color=circle_line_color, fill_color=circle_color,
                   fill_alpha=circle_opacity, line_width=1)

    filt_plot.add_glyph(source, glyph)

    callback = CustomJS(args=dict(source=source, vline=vline), code="""
    var data = source.data;
    var s = cb_obj.value
    var sizes = data['sizes']
    for (var i = 0; i < sizes.length; i++) {
        sizes[i] = s/2
    }
    var vdata = vline.data;
    var step = vdata['s']
    step[0] = s
    vline.change.emit();
    source.change.emit();
    """)

    barcode_plot = figure(title='Barcode',
                          plot_width=800,
                          plot_height=200,
                          min_border=0,
                          aspect_ratio="auto",
                          height_policy="auto",
                          toolbar_location=None,
                          x_axis_label='Filtration Value',
                          x_range=(radius_range[0], radius_range[1]))

    lscape_plot = figure(title='Landscapes',
                         plot_width=800,
                         plot_height=200,
                         min_border=0,
                         aspect_ratio="auto",
                         height_policy="auto",
                         toolbar_location=None,
                         x_axis_label='Filtration Value',
                         x_range=(radius_range[0], radius_range[1]))



    for bar in range(len(H_0_Bars)):
        if H_0_Bars[bar, 1] < radius_range[1]:
            barcode_plot.line([H_0_Bars[bar, 0], H_0_Bars[bar, 1]], [bar / len(H_0_Bars), bar / len(H_0_Bars)],
                              legend_label='H0 Bars', color=H_0_color, line_width=bar_line_width)
        else:
            barcode_plot.line([H_0_Bars[bar, 0], radius_range[1]], [bar / len(H_0_Bars), bar / len(H_0_Bars)],
                              legend_label='H0 Bars', color=H_0_color, line_width=bar_line_width)
    for bar in range(len(H_1_Bars)):
        if H_1_Bars[bar, 1] < radius_range[1]:
            barcode_plot.line([H_1_Bars[bar, 0], H_1_Bars[bar, 1]],
                              [3 / 2 + bar / len(H_1_Bars), 3 / 2 + bar / len(H_1_Bars)],
                              legend_label='H1 Bars', color=H_1_color, line_width=bar_line_width)
        else:
            barcode_plot.line([H_1_Bars[bar, 0], radius_range[1]],
                              [3 / 2 + bar / len(H_1_Bars), 3 / 2 + bar / len(H_1_Bars)],
                              legend_label='H1 Bars', color=H_1_color, line_width=bar_line_width)

    barcode_plot.ray(x="s", y="y", length="y", angle="angle", source=vline, color="#FB8072", line_width=bar_line_width)

    barcode_plot.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    barcode_plot.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    barcode_plot.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    barcode_plot.legend.location = "bottom_right"
    H0rivet_barcode = ripser_to_rivet_bcode(H_0_Bars)
    L = compute_landscapes(H0rivet_barcode)

    for k in range(len(L.landscapes)):
        n = np.shape(L.landscapes[k].critical_points)[0]
        x = L.landscapes[k].critical_points[1:n, 0]
        y = L.landscapes[k].critical_points[1:n, 1]
        if k < 3:
            lscape_plot.line(x=x, y=y, color=H_0_color, line_alpha=1 / (k + 1), line_width=bar_line_width,
                             legend_label='H0: k =' + str(k + 2), muted_color=vertical_line_color, muted_alpha=1)
        else:
            lscape_plot.line(x=x, y=y, color=H_0_color, line_alpha=1 / (k + 1), line_width=bar_line_width)

    H_1_rivet_barcode = ripser_to_rivet_bcode(H_1_Bars)
    L = compute_landscapes(H_1_rivet_barcode)

    for k in range(len(L.landscapes)):
        n = np.shape(L.landscapes[k].critical_points)[0]
        x = L.landscapes[k].critical_points[1:n, 0]
        y = L.landscapes[k].critical_points[1:n, 1]
        if k < 3:
            lscape_plot.line(x=x, y=y, color=H_1_color, line_alpha=1 / (k + 1), line_width=bar_line_width,
                             legend_label='H1: k =' + str(k + 1), muted_color=vertical_line_color, muted_alpha=1)
        else:
            lscape_plot.line(x=x, y=y, color=H_1_color, line_alpha=1 / (k + 1), line_width=bar_line_width)
    lscape_plot.legend.location = "top_right"
    lscape_plot.legend.click_policy = "mute"

    lscape_plot.ray(x="s", y="y", length="y", angle="angle",
                    source=vline, color=vertical_line_color, line_width=bar_line_width)

    slider = Slider(start=radius_range[0], end=radius_range[1], value=radius_range[0],
                    step=(radius_range[1] - radius_range[0]) / 100, title="Rips Parameter")
    slider.js_on_change('value', callback)

    layout = column( row(filt_plot) ,slider,  barcode_plot, lscape_plot, sizing_mode="scale_both")

    return layout
