import yt
import matplotlib.pyplot as plt

c = ([-80, -150, 0], "pc")
sphere = ds.sphere(c, (100.0, "pc"))

plot = yt.PhasePlot(sphere, ("gas", "temperature"), ("gas", "xray_emissivity_0.5_2_keV"), [("gas", "cell_volume")])
plot.set_ylim(1e-33, 1e-20)
plot.set_xlim(1e5, 1e8)
fig = plt.gcf()
fig.set_size_inches(2, 1) 
plot.save('phaseplot_emis_vs_temp.pdf')