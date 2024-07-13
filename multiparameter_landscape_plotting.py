import copy
from typing import List

import numpy as np
from bokeh.layouts import column
from bokeh.layouts import row
from bokeh.models import HoverTool, Range1d, CustomJS, RangeSlider, Rect, LinearColorMapper, Circle, ColorBar, Slider
from bokeh.palettes import inferno
from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure

from scipy.spatial import distance_matrix

from multiparameter_landscape import multiparameter_landscape
from helper_functions import normalise_filter, Compute_Rivet


def compute_mean_landscape(Sample: List[multiparameter_landscape]):
    """
    :param Sample: List of multiparameter landscape
    :return: Multiparameter Landscape Object whose landscape matrix is the mean of the collection
    """
    mean_landscape = copy.copy(Sample[0])
    mean_landscape.landscape_matrix = Sample[0].landscape_matrix
    for s in range(0, len(Sample)):
        mean_landscape.landscape_matrix += Sample[s].landscape_matrix
    mean_landscape.landscape_matrix /= len(Sample)
    return mean_landscape


def landscape_matrix_to_img(landscape_matrix):
    img = np.rot90(landscape_matrix.T)
    return img


def stack_landscapes(Sample):
    landscape_stack = np.zeros((len(Sample),) + Sample[0].landscape_matrix.shape)
    for s in range(len(Sample)):
        landscape_stack[s, :, :, :] = Sample[s].landscape_matrix

    return landscape_stack


def rotate_stack(landscape_stack):
    rotated_landscape_stack = np.zeros(landscape_stack.shape)
    for s in range(landscape_stack.shape[0]):
        for k in range(landscape_stack.shape[1]):
            rotated_landscape_stack[s, k, :, :] = landscape_matrix_to_img(landscape_stack[s, k, :, :])
    return rotated_landscape_stack


def plot_a_two_parameter_landscape(multi_landscape: multiparameter_landscape, index: int, TOOLTIPS=None, high=None,
                                   x_axis_label=None, y_axis_label=None):
    if (index > multi_landscape.maximum_landscape_depth) or (index < 0):
        raise TypeError('Index out of range')

    if TOOLTIPS is None:
        TOOLTIPS = [
            ("x", "$x"),
            ("y", "$y"),
            ("value", "@image")
        ]

    if high is None:
        high = np.max(multi_landscape.landscape_matrix[index, :, :])

    color_mapper = LinearColorMapper(palette=inferno(256), low=0, high=high)

    source = ColumnDataSource(
        data=dict(image=[landscape_matrix_to_img(multi_landscape.landscape_matrix[index, :, :])],
                  x=[multi_landscape.bounds.lower_left[0]],
                  y=[multi_landscape.bounds.lower_left[1]],
                  dw=[multi_landscape.bounds.upper_right[0] - multi_landscape.bounds.lower_left[0]],
                  dh=[multi_landscape.bounds.upper_right[1] - multi_landscape.bounds.lower_left[1]])
    )

    plot = figure(
        x_range=Range1d(multi_landscape.bounds.lower_left[0], multi_landscape.bounds.upper_right[0], bounds='auto'),
        y_range=Range1d(multi_landscape.bounds.lower_left[1], multi_landscape.bounds.upper_right[1], bounds='auto'),
        title="Multiparameter Landscape k=" + str(index + 1),
        width=250,
        height=250,
        match_aspect=True,
        toolbar_location=None
    )

    if x_axis_label is not None:
        plot.xaxis.axis_label = x_axis_label
    if y_axis_label is not None:
        plot.yaxis.axis_label = y_axis_label

    img = plot.image(source=source.data, image='image', x='x', y='y',
                     dw='dw', dh='dh',
                     color_mapper=color_mapper)

    img_hover = HoverTool(renderers=[img], tooltips=TOOLTIPS)
    plot.add_tools(img_hover)

    return plot


def plot_multiparameter_landscapes(multi_landscape: multiparameter_landscape, indices=None,
                                   TOOLTIPS=None, normalise_scale=True, x_axis_label=None, y_axis_label=None):
    """ Plots the landscapes in the collection to a single axes in range index = [ind_min,ind_max]"""

    if indices is None:
        indices = [1, multi_landscape.maximum_landscape_depth]
    elif type(indices[0]) is int and type(indices[1]) is int and \
            (indices[1] <= multi_landscape.maximum_landscape_depth):
        indices = indices
    else:
        raise TypeError('Index range must provide integer values no bigger that the landscape depth')
    if normalise_scale:
        high = np.max(multi_landscape.landscape_matrix[indices[0] - 1, :, :])
    else:
        high = None
    # Make Plot #

    plots = []

    for k in range(indices[0] - 1, indices[1]):
        plots.append(plot_a_two_parameter_landscape(multi_landscape, k, TOOLTIPS=TOOLTIPS, high=high,
                                                    x_axis_label=x_axis_label, y_axis_label=y_axis_label))

    for k in range(0, len(plots)):
        plots[k].x_range = plots[0].x_range
        plots[k].y_range = plots[0].y_range

    # Layout Plot #
    DOM = row(plots)

    return DOM


def compare_multiparameter_landscape_samples(Samples: List[List[multiparameter_landscape]], indices=None,
                                             TOOLTIPS=None, GroupLabels: List[str] = None,
                                             colors: List[str] = None):
    """
    Plots the mean landscapes of each collection to a single axes in range index = [ind_min,ind_max]
    together with a boxplot comparing the distribution of the functional values yielded by integrating the landscapes
    over a user specified range


    :param Samples: List
    List of lists of multiparameter_landscape objects so Sample[k][i] contains the
    i^th multiparameter landscape from the k^th Sample Group. All multiparameter
    landscapes are assumed to have the same bounds as Samples[0][0].
    :param indices:
    The range of indices to be plotted
    :param TOOLTIPS:
    A bokeh TOOLTIPS list to control features that can be probed in the plot
    :param GroupLabels:
    A list of the names of the Sample Groups
    :param colors:
    A list of the colors for the Sample Groups
    """

    for k in range(len(Samples)):
        if not (Samples[k][0].bounds.lower_left == Samples[0][0].bounds.lower_left
                and Samples[k][0].bounds.upper_right == Samples[0][0].bounds.upper_right):
            raise TypeError('Inconsistent bounds for landscapes in Sample List')
        for i in range(len(Samples[k])):
            if not (Samples[k][i].bounds.lower_left == Samples[0][0].bounds.lower_left
                    and Samples[k][i].bounds.upper_right == Samples[0][0].bounds.upper_right):
                raise TypeError('Inconsistent bounds for landscapes in Sample List')

    for k in range(len(Samples)):
        if not (Samples[k][0].weight == Samples[0][0].weight).all():
            raise TypeError('Inconsistent weights for landscapes in Sample List')
        for i in range(len(Samples[k])):
            if not (Samples[k][i].weight == Samples[k][0].weight).all():
                raise TypeError('Inconsistent weights for landscapes in Sample List')

    for k in range(len(Samples)):
        if not Samples[k][0].grid_step_size == Samples[0][0].grid_step_size:
            raise TypeError('Inconsistent grid step size for landscapes in Sample List')
        for i in range(len(Samples[k])):
            if not Samples[k][i].grid_step_size == Samples[k][0].grid_step_size:
                raise TypeError('Inconsistent grid step size for landscapes in Sample List')

    if indices is None:
        indices = [1, Samples[0][0].maximum_landscape_depth]
    elif type(indices[0]) is int and type(indices[1]) is int and \
            (indices[1] <= Samples[0][0].maximum_landscape_depth):
        indices = indices
    else:
        raise TypeError('Index range must provide integer values no bigger that the landscape depth')

    if TOOLTIPS is None:
        TOOLTIPS = [
            ("x", "$x"),
            ("y", "$y"),
            ("value", "@image")
        ]

    if GroupLabels is None:
        GroupLabels = ['Group ' + str(s + 1) for s in range(len(Samples))] * (indices[1] - indices[0] + 1)
    else:
        if not len(GroupLabels) == len(Samples):
            raise TypeError('GroupLabels list is not the same length as the sample list')
        GroupLabels = GroupLabels * (indices[1] - indices[0] + 1)

    if colors is None:
        colors = ['plum', 'powderblue', 'gold', 'greenyellow', 'mediumblue', 'firebrick']
    else:
        if not len(colors) == len(Samples):
            raise TypeError('Color list is not the same length as the sample list')

    bounds = Samples[0][0].bounds
    grid_step_size = Samples[0][0].grid_step_size
    weight = Samples[0][0].weight

    ######################

    # Define ColumnDataSources #

    rectangle_source = ColumnDataSource(
        data=dict(
            x=[(bounds.lower_left[0] + bounds.upper_right[0]) / 2],
            y=[(bounds.lower_left[1] + bounds.upper_right[1]) / 2],
            width=[bounds.upper_right[0] - bounds.lower_left[0]],
            height=[bounds.upper_right[1] - bounds.lower_left[1]])
    )

    rect = Rect(
        x='x',
        y='y',
        width='width',
        height='height',
        fill_alpha=0.2,
        fill_color='white',
        line_color='white',
    )

    landscape_properties = ColumnDataSource(
        data=dict(
            bounds=[[bounds.lower_left, bounds.upper_right]],
            x_steps=[
                int(round((bounds.upper_right[0] - bounds.lower_left[0]) / grid_step_size * weight[1] + 1))],
            y_steps=[
                int(round((bounds.upper_right[1] - bounds.lower_left[1]) / grid_step_size * weight[0] + 1))],
            indices=[indices]

        )
    )

    boxplot_source = ColumnDataSource(
        data=dict(
            cats=["Group " + str(s + 1) + ", k=" + str(k + 1) for k in range(indices[0] - 1, indices[1])
                  for s in range(len(Samples))],
            colors=[colors[s] for _ in range(indices[0] - 1, indices[1]) for s in range(len(Samples))],
            q1=[-1] * (len(Samples) * (indices[1] - indices[0] + 1)),
            q2=[0] * (len(Samples) * (indices[1] - indices[0] + 1)),
            q3=[1] * (len(Samples) * (indices[1] - indices[0] + 1)),
            lower=[-2] * (len(Samples) * (indices[1] - indices[0] + 1)),
            upper=[2] * (len(Samples) * (indices[1] - indices[0] + 1)),
            labels=GroupLabels
        )
    )

    ######################

    # Callback Functions #

    rotated_samples = [rotate_stack(stack_landscapes(Samples[s])) for s in range(len(Samples))]

    callback_x = CustomJS(args=dict(rectangle_source=rectangle_source, lp=landscape_properties, bp=boxplot_source,
                                    samples=rotated_samples), code="""
        
        function Find_Median(list) {
          var Size = list.length;
          list = Array.from(list).sort(function(a, b) {
            return a - b
          });
          console.log('sorted')
          console.log(list)
          var Final_Number = 0;
          var HalfWay = 0;
          if (Size % 2 == 0) {
            HalfWay = Math.round((list.length) / 2)-1;
            console.log('Halfway index')
            console.log(HalfWay)
            var Value1 = list[HalfWay];
            var Value2 = list[HalfWay + 1];
            console.log('Value1')
            console.log(Value1)
            console.log('Value2')
            console.log(Value2)
            var Number = Value1 + Value2;
            Final_Number = Number / 2;
            console.log('Final_Number')
            console.log(Final_Number)
            
          } else {
            HalfWay = Math.round(list.length / 2)-1;
            console.log('Halfway index')
            console.log(HalfWay)
            Final_Number = list[HalfWay];
            console.log('Final_Number')
            console.log(Final_Number)
          }
          console.log('Final_Number')
          console.log(Final_Number)
          return Final_Number;
        
        }
        
        function BiggerElements(val)
       {
         return function(evalue, index, array)
         {
         return (evalue >= val);
         };
       }
       
       function SmallerElements(val)
       {
         return function(evalue, index, array)
         {
         return (evalue <= val);
         };
       }
        
        var data = rectangle_source.data;
        var xrange = cb_obj.value
        data['x'] = [(xrange[0]+xrange[1])/2]
        data['width'] = [(xrange[1]-xrange[0])]
        rectangle_source.change.emit();

        var bpdata = bp.data;

        var ylow = data['y'][0] - data['height'][0]/2;
        var yhigh = data['y'][0] + data['height'][0]/2;
        var a = Math.round(Math.PI)
        
        var bounds = lp.data['bounds'][0];
        var x_steps = lp.data['x_steps'];
        var y_steps = lp.data['y_steps'];
        var indices = lp.data['indices'][0];

        var xmin = Math.round(xrange[0]/ bounds[1][0]* x_steps[0]);
        var xmax = Math.round(xrange[1]/ bounds[1][0]* x_steps[0]);
        var ymin = Math.round(ylow/ bounds[1][1]* y_steps[0]);
        var ymax = Math.round(yhigh/ bounds[1][1]* y_steps[0]);

        var q1 = bpdata['q1']
        var q2 = bpdata['q2']
        var q3 = bpdata['q3']
        var lower = bpdata['lower']
        var upper = bpdata['upper']
        
        var number_of_samples = samples.length
        console.log("number_of_samples")
        console.log(number_of_samples)
        
        console.log(samples[0])
        console.log(samples[0].length)
        
        for (var s = 0; s < number_of_samples; s++ )
        {   
            for (var k = 0; k < indices[1]-indices[0]+1 ; k++ )
            {   
                console.log("Index")
                console.log(k)
                console.log('samples[s].length')
                console.log(samples[s].length)
                var holder = Array.from(Array(samples[s].length), () => 0)
                console.log('holder')
                console.log(holder)
                for (var l = 0; l < samples[s].length; l++ )
                {
                console.log("Sample")
                console.log(l)
                var value = 0
                    for (var i = xmin; i < xmax; i++)
                    {
                        for (var j = ymin; j < ymax ; j++)
                        {   
                            value += samples[s][l][k][j][i]
                        }
                    }
                console.log('value')
                console.log(value)
                holder[l] = value  / (x_steps * y_steps)
                }
                console.log('holder')
                console.log(holder)
                console.log('Median')
                console.log(Find_Median(holder))
                console.log('Median^')
                q2[number_of_samples*k + s] = Find_Median(holder)
                q1[number_of_samples*k + s] = Find_Median(holder.filter(SmallerElements(Find_Median(holder))))
                q3[number_of_samples*k + s] = Find_Median(holder.filter(BiggerElements(Find_Median(holder))))
                lower[number_of_samples*k + s] = Math.min.apply(Math,holder)
                upper[number_of_samples*k + s] = Math.max.apply(Math,holder)
            }

        }
        console.log('q2')
        console.log(q2)
        bp.change.emit()
    """)

    callback_y = CustomJS(args=dict(rectangle_source=rectangle_source, lp=landscape_properties, bp=boxplot_source,
                                    samples=rotated_samples), code="""
        function Find_Median(list) {
          var Size = list.length;
          list = Array.from(list).sort(function(a, b) {
            return a - b
          });
          console.log('sorted')
          console.log(list)
          var Final_Number = 0;
          var HalfWay = 0;
          if (Size % 2 == 0) {
            HalfWay = Math.round((list.length) / 2)-1;
            console.log('Halfway index')
            console.log(HalfWay)
            var Value1 = list[HalfWay];
            var Value2 = list[HalfWay + 1];
            console.log('Value1')
            console.log(Value1)
            console.log('Value2')
            console.log(Value2)
            var Number = Value1 + Value2;
            Final_Number = Number / 2;
            console.log('Final_Number')
            console.log(Final_Number)
            
          } else {
            HalfWay = Math.round(list.length / 2)-1;
            console.log('Halfway index')
            console.log(HalfWay)
            Final_Number = list[HalfWay];
            console.log('Final_Number')
            console.log(Final_Number)
          }
          console.log('Final_Number')
          console.log(Final_Number)
          return Final_Number;
        
        }
        
        function BiggerElements(val)
       {
         return function(evalue, index, array)
         {
         return (evalue >= val);
         };
       }
       
       function SmallerElements(val)
       {
         return function(evalue, index, array)
         {
         return (evalue <= val);
         };
       }                            
        
        var data = rectangle_source.data;
        var yrange = cb_obj.value
        data['y'] = [(yrange[0]+yrange[1])/2]
        data['height'] = [(yrange[1]-yrange[0])]
        rectangle_source.change.emit();

        var bpdata = bp.data;
        
        var bounds = lp.data['bounds'][0];
        var x_steps = lp.data['x_steps'];
        var y_steps = lp.data['y_steps'];
        var indices = lp.data['indices'][0];
        
        var xlow = data['x'][0] - data['width'][0]/2;
        var xhigh = data['x'][0] + data['width'][0]/2;
        var x_steps = lp.data['x_steps'];
        var y_steps = lp.data['y_steps'];

        var xmin = Math.round(xlow/ bounds[1][0]* x_steps[0]);
        var xmax = Math.round(xhigh/ bounds[1][0]* x_steps[0]);
        var ymin = Math.round(yrange[0]/ bounds[1][1]* y_steps[0]);
        var ymax = Math.round(yrange[1]/ bounds[1][1]* y_steps[0]);

        var q1 = bpdata['q1']
        var q2 = bpdata['q2']
        var q3 = bpdata['q3']
        var lower = bpdata['lower']
        var upper = bpdata['upper']
        var number_of_samples = samples.length
        console.log("number_of_samples")
        console.log(number_of_samples)
        
        console.log(samples[0])
        console.log(samples[0].length)
        
        for (var s = 0; s < number_of_samples; s++ )
        {   
            for (var k = 0; k < indices[1]-indices[0]+1 ; k++ )
            {   
                console.log("Index")
                console.log(k)
                console.log('samples[s].length')
                console.log(samples[s].length)
                var holder = Array.from(Array(samples[s].length), () => 0)
                console.log('holder')
                console.log(holder)
                for (var l = 0; l < samples[s].length; l++ )
                {
                console.log("Sample")
                console.log(l)
                var value = 0
                    for (var i = xmin; i < xmax; i++)
                    {
                        for (var j = ymin; j < ymax ; j++)
                        {   
                            value += samples[s][l][k][j][i]
                        }
                    }
                console.log('value')
                console.log(value)
                holder[l] = value  / (x_steps * y_steps)
                }
                console.log('holder')
                console.log(holder)
                console.log('Median')
                console.log(Find_Median(holder))
                console.log('Median^')
                q2[number_of_samples*k + s] = Find_Median(holder)
                q1[number_of_samples*k + s] = Find_Median(holder.filter(SmallerElements(Find_Median(holder))))
                q3[number_of_samples*k + s] = Find_Median(holder.filter(BiggerElements(Find_Median(holder))))
                lower[number_of_samples*k + s] = Math.min.apply(Math,holder)
                upper[number_of_samples*k + s] = Math.max.apply(Math,holder)
            }

        }
        console.log('q2')
        console.log(q2)
        bp.change.emit()
    """)

    #########################

    # Widgets and callbacks #

    x_range_slider = RangeSlider(start=bounds.lower_left[0], end=bounds.upper_right[0],
                                 value=(bounds.lower_left[0], bounds.upper_right[0]), step=.1, title="x_range",
                                 width=300)
    y_range_slider = RangeSlider(start=bounds.lower_left[1], end=bounds.upper_right[1],
                                 value=(bounds.lower_left[1], bounds.upper_right[1]), step=.1, title="y_range",
                                 width=300)
    x_range_slider.js_on_change('value', callback_x)
    y_range_slider.js_on_change('value', callback_y)

    #########################

    # Make Plots #

    mean_landscape_plots = []

    for s in range(len(Samples)):
        mean_landscape = compute_mean_landscape(Samples[s])
        row_of_plots = plot_multiparameter_landscapes(mean_landscape, indices, TOOLTIPS=TOOLTIPS)
        row_of_plots.sizing_mode = "scale_both"
        mean_landscape_plots.append(row_of_plots)
        for plot in row_of_plots.children:
            plot.add_glyph(rectangle_source, rect)
            plot.plot_height = 250
            plot.plot_width = 250
            plot.border_fill_color = colors[s]
            plot.min_border = 15

    boxplots = figure(x_range=boxplot_source.data['cats'], height=250)

    # stems
    boxplots.segment(source=boxplot_source, x0='cats', y0='upper', x1='cats', y1='q3', color="black")
    boxplots.segment(source=boxplot_source, x0='cats', y0='lower', x1='cats', y1='q1', color="black")

    # boxes
    boxplots.vbar(source=boxplot_source, x='cats', top='q3',
                  bottom='q2', width=0.5, fill_color='colors', line_color='black', legend_group='labels')
    boxplots.vbar(source=boxplot_source, x='cats', top='q2',
                  bottom='q1', width=0.5, fill_color='colors', line_color='black')

    # whiskers
    boxplots.rect(source=boxplot_source, x='cats', y='lower', width=0.2, height=0.000001, color='black')
    boxplots.rect(source=boxplot_source, x='cats', y='upper', width=0.2, height=0.000001, color='black')

    boxplots.legend.glyph_width = 50
    boxplots.legend.glyph_height = 20
    boxplots.legend.label_text_font_size = '10pt'

    boxplots.xgrid.grid_line_color = None
    boxplots.sizing_mode = "scale_both"

    #         ##########################
    #         # Layout and Show Plot #

    sliders = column([x_range_slider, y_range_slider])
    DOM = column(column(mean_landscape_plots), sliders, boxplots)

    return DOM


def Rips_Filter_Bifiltration(filtered_points, radius_range, palette="Viridis256", FilterName="Filter",
                             maxind: int = None, dim: int = None):
    if maxind is None:
        maxind = 5
    if dim is None:
        dim = 0

    points = filtered_points[:, :2]
    filter = filtered_points[:, 2]

    alpha = np.ones(filter.shape) * 0.3
    exp_cmap = LinearColorMapper(palette=palette,
                                 low=radius_range[0],
                                 high=radius_range[1])

    source = ColumnDataSource(data=dict(x=points[:, 0],
                                        y=points[:, 1],
                                        sizes=(radius_range[0] + radius_range[1]) / 4 * np.ones(points.shape[0]),
                                        filter=filter,
                                        alpha=alpha

                                        )
                              )

    vline = ColumnDataSource(data=dict(c=[radius_range[0]],
                                       y=[0],
                                       angle=[np.pi / 2]
                                       )
                             )
    hline = ColumnDataSource(data=dict(s=[radius_range[0]],
                                       x=[0],
                                       angle=[0]
                                       )
                             )

    filter_plot = figure(title='Filtration',
                         plot_width=360,
                         plot_height=430,
                         min_border=0,
                         toolbar_location=None,
                         match_aspect=True)

    glyph = Circle(x="x", y="y", radius="sizes", line_color="black",
                   fill_color={'field': 'filter', 'transform': exp_cmap},
                   fill_alpha="alpha", line_width=1, line_alpha="alpha")

    filter_plot.add_glyph(source, glyph)
    filter_plot.add_layout(ColorBar(color_mapper=exp_cmap, location=(0, 0), orientation="horizontal"), "below")

    rips_callback = CustomJS(args=dict(source=source, hline=hline), code="""
    var data = source.data;
    var s = cb_obj.value
    var sizes = data['sizes']


    for (var i = 0; i < sizes.length; i++) {
        sizes[i] = s/2
    }
    var hdata = hline.data;
    var step = hdata['s']
    step[0] = s
    hline.change.emit();
    source.change.emit();
    """)

    filter_callback = CustomJS(args=dict(source=source, vline=vline), code="""
    var data = source.data;
    var c = cb_obj.value
    var alpha = data['alpha']
    var filter = data['filter']

    for (var i = 0; i<filter.length; i++){
        if(filter[i]>c){
        alpha[i] = 0
        }
        if(filter[i]<=c){
        alpha[i] = 0.3
        }
    }

    var vdata = vline.data;
    var step = vdata['c']
    step[0] = c
    vline.change.emit();
    source.change.emit();
    """)

    rips_slider = Slider(start=radius_range[0], end=radius_range[1], value=(radius_range[0] + radius_range[1]) / 2,
                         step=(radius_range[1] - radius_range[0]) / 100, title="Rips",
                         orientation="vertical",
                         height=300,
                         direction="rtl",
                         margin=(10, 40, 10, 60)
                         )

    rips_slider.js_on_change('value', rips_callback)

    filter_slider = Slider(start=radius_range[0], end=radius_range[1], value=radius_range[1],
                           step=(radius_range[1] - radius_range[0]) / 100, title=FilterName,
                           orientation="horizontal",
                           aspect_ratio=10,
                           width_policy="auto",
                           direction="ltr", width=300,
                           margin=(10, 10, 10, 40)
                           )

    filter_slider.js_on_change('value', filter_callback)

    # Run Rivet and Landscape Computation #

    computed_data = Compute_Rivet(filtered_points, resolution=50, dim=dim, RipsMax=radius_range[1])
    multi_landscape = multiparameter_landscape(computed_data,
                                               grid_step_size=(radius_range[1] - radius_range[0]) / 100,
                                               bounds=[[radius_range[0], radius_range[0]],
                                                       [radius_range[1], radius_range[1]]],
                                               maxind=maxind)

    TOOLTIPS = [
        (FilterName, "$x"),
        ("Radius", "$y"),
        ("Landscape_Value", "@image")
    ]

    landscape_plots = plot_multiparameter_landscapes(multi_landscape, indices=[1, maxind], TOOLTIPS=TOOLTIPS,
                                                     x_axis_label=FilterName, y_axis_label="Rips Parameter")

    for plot in landscape_plots.children:
        plot.ray(x="c", y="y", length="y", angle="angle",
                 source=vline, color='white', line_width=2, alpha=0.5)
        plot.ray(x="x", y="s", length="x", angle="angle",
                 source=hline, color='white', line_width=2, alpha=0.5)

    layout = column(row(rips_slider, column(filter_plot, filter_slider)), row(landscape_plots),
                    sizing_mode="scale_both")

    return layout


def Rips_Codensity_Bifiltration(points, radius_range, kNN: int = None, maxind: int = None, dim: int = None):
    if kNN is None:
        kNN = 5
    if maxind is None:
        maxind = 5
    if dim is None:
        dim = 0

    D = distance_matrix(points, points)
    sortedD = np.sort(D)
    codensity = np.sum(sortedD[:, :kNN + 1], axis=1)
    codensity = (radius_range[1] - radius_range[0]) * normalise_filter(codensity, 5) + radius_range[0]
    filtered_points = np.hstack((points, np.expand_dims(codensity, axis=1)))

    return Rips_Filter_Bifiltration(filtered_points, radius_range, palette="Viridis256",
                                    FilterName=str(kNN) + "NN-Codensity",
                                    maxind=maxind, dim=dim)
