import os
from helper_functions import sample_circle_disc
from bokeh.plotting import output_file, show
from multiparameter_landscape_plotting import Rips_Codensity_Bifiltration
import multiparameter_landscape_plotting
import rivet
from multiparameter_landscape import multiparameter_landscape

output_file("test.html")

rivet_location = '/home/ollie/Documents/RIVET/MarchReinstall/rivet'  # Might be in a different location for you
os.listdir(rivet_location + '/data/Test_Point_Clouds')

points = 0.5 * sample_circle_disc(0.8, 100)
# show(Rips_Codensity_Bifiltration(points=points, radius_range=[0, 1]))

      # Stats Plot Test

computed_file_path = rivet.compute_file(
    rivet_location + '/data/Test_Point_Clouds/circle_data_60pts_codensity.txt',
    homology=0,
    x=40,
    y=40)

with open(computed_file_path, 'rb') as f:
    computed_data = f.read()

computed_file_path = rivet.compute_file(
    rivet_location + '/data/Test_Point_Clouds/circle_data_60pts_codensity.txt',
    homology=1,
    x=40,
    y=40)

with open(computed_file_path, 'rb') as f:
    computed_data2 = f.read()

computed_file_path = rivet.compute_file(
    rivet_location + '/data/Test_Point_Clouds/circle_data_60pts_codensity.txt',
    homology=0,
    x=10,
    y=10)

with open(computed_file_path, 'rb') as f:
    computed_data3 = f.read()

multi_landscape = multiparameter_landscape(computed_data, grid_step_size=0.5, bounds=[(0, 0), (10, 10)], weight=[1, 1])
multi_landscape2 = multiparameter_landscape(computed_data2, grid_step_size=0.5, bounds=[(0, 0), (10, 10)],
                                            weight=[1, 1])
multi_landscape3 = multiparameter_landscape(computed_data3, grid_step_size=0.5, bounds=[(0, 0), (10, 10)],
                                            weight=[1, 1])
show(multiparameter_landscape_plotting.compare_multiparameter_landscape_samples(
    [[multi_landscape, multi_landscape2, multi_landscape3], [multi_landscape, multi_landscape3],
     [multi_landscape2, multi_landscape3]],
    indices=[1, 5]))
