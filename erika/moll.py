import os
import pickle
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

pickle_dir = "/home/erea/data/B6-1pc/data-float32"
output_dir = "/home/erea/data/B6-1pc/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pickle_files_to_include = [
    "SILCC_hdf5_plt_cnt_1080-r0150-c000050-data-float32.pkl",
    "SILCC_hdf5_plt_cnt_1080-r0150-c000100-data-float32.pkl",
    "SILCC_hdf5_plt_cnt_1080-r0150-c000167-data-float32.pkl",
    "SILCC_hdf5_plt_cnt_1080-r0150-c000500-data-float32.pkl",
    "SILCC_hdf5_plt_cnt_1080-r0150-c001000-data-float32.pkl",
    "SILCC_hdf5_plt_cnt_1080-r0150-c001670-data-float32.pkl"
]

colorbar_labels = {
    "coldens": "log(Column Density ($\\mathrm{cm}^{-2}$))",
    "coldens2": "log(Column Density ($\\mathrm{cm}^{-2}$))",
    "Xraylum": "log(X-ray lum (erg/s))",
    "Xrayflx": "log(X-ray Flux (erg/s/sr))",
    "emmeasure": "log(Emission Measure ($\\mathrm{cm}^{-5}$))",
    "radius": "Radius (pc)",
    "bubble_open": "Open fraction of the bubble"
}

field_limits = {
    "coldens": (None, None),    
    "coldens2": (-7, -1),
    "Xraylum": (18, 32),        
    "Xrayflx": (18, 32),
    "emmeasure": (-3, 3),
    "radius": (40, 160),
    "bubble_open": (None, None)
}

color_maps = {
    "coldens": "viridis",
    "coldens2": "plasma",
    "Xraylum": "inferno",
    "Xrayflx": "magma",
    "emmeasure": "afmhot",
    "radius": "coolwarm",
    "bubble_open": "coolwarm"
}

def extract_info_from_filename(filename):
    parts = filename.split('-')
    
    print(f"Filename parts: {parts}")

    if len(parts) < 4:
        return None, None

    try:
        time_part = parts[0].split('_')[-1]
        density_part = parts[2].replace('c', '')

        print(f"Extracted time part: {time_part}")
        print(f"Extracted density part: {density_part}")
        
        time_in_myr = float(time_part) / 100
        
        density = int(density_part) / 1e6
        
        return time_in_myr, density
    except (IndexError, ValueError) as e:
        print(f"Error extracting info: {e}")
        return None, None

def create_mollweide_projections(pickle_file_path, output_dir):
    base_name = os.path.basename(pickle_file_path).replace('.pkl', '')

    # time and col density values
    time_in_myr, density = extract_info_from_filename(base_name)
    
    print(f"Extracted time: {time_in_myr} Myr")
    print(f"Extracted density: {density:.2e}")

    if time_in_myr is None or density is None:
        print(f"Nome del file non contiene le informazioni richieste: {base_name}")
        return

    # Load pickle file
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)

    fields_to_plot = [
        ("coldens", True),
        ("coldens2", True),
        ("Xraylum", True),
        ("Xrayflx", True),
        ("emmeasure", True),
        ("radius", False),
        ("bubble_open", False)
    ]

    for name, norm in fields_to_plot:
        field = data[name]
        fig = plt.figure(figsize=(8, 3))
        if norm:
            plt_field = np.log10(field).copy()
        else:
            plt_field = field.copy()

        vmin, vmax = field_limits.get(name, (None, None))

        hp.mollview(
            plt_field,
            fig=fig.number,
            return_projected_map=True,
            title=f'Time: {time_in_myr:.1f} Myr - Column Density: {density:.2e} - {name}',
            cbar=False,
            min=vmin,
            max=vmax,
            cmap=color_maps.get(name, 'viridis')
        )
        hp.graticule()

        ax = plt.gca()
        image = ax.get_images()[0]
        cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.8)

        cbar_label = colorbar_labels.get(name, name)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)

        output_file_path = os.path.join(output_dir, f'{base_name}_{name}.pdf')
        plt.savefig(output_file_path, bbox_inches="tight")
        plt.close()

pickle_files = [f for f in os.listdir(pickle_dir) if f in pickle_files_to_include]

for pickle_file in pickle_files:
    pickle_file_path = os.path.join(pickle_dir, pickle_file)
    print(f"Generazione dei grafici per {pickle_file_path}")
    create_mollweide_projections(pickle_file_path, output_dir)
