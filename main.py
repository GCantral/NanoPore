
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import HelperFunctions as hf
import graphical_analysis as ga
import math






def make_pc1_density_video(
    dat_path,
    out_path="pc1_density.mp4",
    bins=100,
    normalize=True,
    fps=15,
    figsize=(7, 4),
    title="Molecular Density",
    ylim=None,
    smooth_sigma_bins=None,
    time_format="t = {t:.2f}",
):

    print("Computing PC1 densities")
    res = hf.compute_z_density(
        dat_path,
        bins=bins,
        average_frames=False,
        normalize=normalize,
    )

    s_centers = res["z_centers"]
    profiles = res["density"]
    times = res.get("times", None) or [None] * len(profiles)

    if not isinstance(profiles, (list, tuple)):
        raise RuntimeError("Expected per-frame densities")

    n_frames = len(profiles)
    if n_frames == 0:
        raise ValueError("dat file is empty")


    print("Plotting Density Graphs")
    if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
        profiles = [ hf.gaussian_smooth(p, smooth_sigma_bins) for p in profiles ]

    if ylim is None:
        ymin = min(float(np.nanmin(p)) for p in profiles)
        ymax = max(float(np.nanmax(p)) for p in profiles)
        span = ymax - ymin
        if span <= 0:
            span = 1.0 if ymax == 0 else abs(ymax) * 0.1
        pad = 0.05 * span
        ylim = (ymin - pad, ymax + pad)

    ylab = "Linear density [1 / nm]" if normalize else "Counts per bin"


    fig, ax = plt.subplots(figsize=figsize)
    (line,) = ax.plot(s_centers, profiles[0], lw=2)
    ax.set_xlim(s_centers.min(), s_centers.max())
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1 coordinate (nm)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.6)

    time_txt = ax.text(
        0.98, 0.95,
        "",
        transform=ax.transAxes,
        ha="right", va="top",
    )

    def _format_time(t, i):
        if t is None:
            return f"frame {i}"
        try:
            return time_format.format(t=float(t))
        except Exception:
            return f"t = {t}"

    def init():
        line.set_data(s_centers, profiles[0])
        time_txt.set_text(_format_time(times[0], 0))
        return (line, time_txt)

    def update(i):
        line.set_data(s_centers, profiles[i])
        time_txt.set_text(_format_time(times[i], i))
        return (line, time_txt)

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, blit=True, interval=1000/fps)

    ext = os.path.splitext(out_path.lower())[1]
    if ext == ".mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2400, extra_args=["-vcodec", "libx264"])
        except Exception as e:
            raise RuntimeError(
                "Saving MP4 requires FFmpeg."
            ) from e
        anim.save(out_path, writer=writer, dpi=150)

    elif ext == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=150)
    else:
        raise ValueError("out_path must end with .mp4 or .gif")

    plt.close(fig)
    print("Complted Video")
    return out_path


def plot_average_profile_after_frame(dat_path,
    start_frame=None,
    start_time=None,
    bins=100,
    normalize=True,
    ax=None,
    figsize=(7, 4),
    savepath=None,
    title="Average PC1 density after threshold",
    show_before=False,
    smooth_sigma_bins=None,
):

    res = hf.compute_z_density(
        dat_path,
        bins=bins,
        average_frames=False,
        normalize=normalize,
    )
    centers = res["z_centers"]
    dens    = res["density"]
    times   = res.get("times", None)
    nF      = len(dens)
    if nF == 0:
        raise ValueError("No frames found.")


    use_time = start_time is not None and times and any(t is not None for t in times)
    if use_time:
        tvals = [(-np.inf if t is None else float(t)) for t in times]
        sel_mask = np.array([t >= float(start_time) for t in tvals], dtype=bool)
    else:
        if start_frame is None:
            start_frame = 0
        si = int(np.clip(start_frame, 0, nF - 1))
        sel_mask = np.zeros(nF, dtype=bool)
        sel_mask[si:] = True

    sel_idx = np.where(sel_mask)[0]
    if sel_idx.size == 0:
        raise ValueError("No frames satisfy the threshold (check start_frame/start_time).")

    stack = np.vstack([dens[i] for i in sel_idx])
    if smooth_sigma_bins:
        stack = np.vstack([hf.gaussian_smooth(row, smooth_sigma_bins) for row in stack])
    avg_after = stack.mean(axis=0)

    avg_before = None
    if show_before and (~sel_mask).any():
        pre_idx = np.where(~sel_mask)[0]
        pre_stack = np.vstack([dens[i] for i in pre_idx])
        if smooth_sigma_bins:
            pre_stack = np.vstack([hf.gaussian_smooth(row, smooth_sigma_bins) for row in pre_stack])
        avg_before = pre_stack.mean(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)



    if(len(path.split("/")[-2].split("_"))>3):label = "No Hairpin"
    else:label = "Hairpin"

    print(label)
    ax.plot(centers, avg_after, lw=2.5, label=label)

    if avg_before is not None:
        lbl_before = f"Average before {'t' if use_time else 'frame'} threshold"
        ax.plot(centers, avg_before, lw=2, alpha=0.7, label=lbl_before)

    ax.set_xlabel("PC1 coordinate (nm)")
    ax.set_ylabel("Linear density [1 / nm]" if normalize else "Counts per bin")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.6)
    if avg_before is not None or len(ax.lines)>1:
        ax.legend(frameon=False)


    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)

    return {
        "centers": centers,
        "avg_after": avg_after,
        "avg_before": avg_before,
        "selected_indices": sel_idx,
        "selected_times": [times[i] if times else None for i in sel_idx],
        "ax": ax,
    }



def pca_coordinates_per_frame(dat_path,
    align_directions=True,
    enforce_right_handed=True,
    whiten=False,
):
    times, centroids, components, eigenvalues, ratios, coords_pca = [], [], [], [], [], []
    prev_U = None


    with open(dat_path, "r") as f:
        for t, R in hf.parse_frame_positions(f):
            N = R.shape[0]

            c = R.mean(axis=0)
            X = R - c
            C = np.cov(X.T, bias=False)
            evals, evecs = np.linalg.eigh(C)
            idx = np.argsort(evals)[::-1]
            lam = evals[idx]
            U = evecs[:, idx]


            if prev_U is not None and align_directions:
                for j in range(3):
                    if np.dot(U[:, j], prev_U[:, j]) < 0:
                        U[:, j] *= -1.0


            if enforce_right_handed and np.linalg.det(U) < 0:
                U[:, 2] *= -1.0

            Xc = R - c
            S = Xc @ U
            if whiten:
                eps = 1e-12
                scale = 1.0 / np.sqrt(np.maximum(lam, eps))
                S = S * scale
            total = float(lam.sum())
            r = lam / total if total > 0 else np.zeros_like(lam)

            times.append(t)
            centroids.append(c)
            components.append(U)
            eigenvalues.append(lam)
            ratios.append(r)
            coords_pca.append(S)

            prev_U = U

    return {
        "coords_pca": coords_pca,
        "times": times,
        "centroids": centroids,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_ratio": ratios,
    }



def plot_spheres_raw(path):


    res = pca_coordinates_per_frame(path+"trajectories/trajectory_z.dat")
    min_max = hf.find_global_extrema(res["coords_pca"])
    coords = res["coords_pca"]

    print("Global maxima [PC1, PC2, PC3]:", min_max["max_xyz"])
    print("Global minimum:", min_max["min_xyz"])
    x_dist = math.ceil(min_max["max_xyz"][0] - min_max["min_xyz"][0])
    y_dist = math.ceil(min_max["max_xyz"][1] - min_max["min_xyz"][1])
    z_dist = math.ceil(min_max["max_xyz"][2] - min_max["min_xyz"][2])

    frame = 233
    distance = np.zeros((x_dist, 100, 100))
    #for i in range(math.ceil(np.abs(min_max["min_xyz"][0])),x_dist):
    for i in range(45,77):
        print(i)
        for j in range(100):
            for k in range(100):
                distance[ i, j, k] = hf.findMin(np.array([min_max["min_xyz"][0]+i, -25+j/2,-25+k/2]),coords[frame])

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    for i in range(math.ceil(np.abs(min_max["min_xyz"][0])), x_dist):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-25,25)
        ax.set_ylim(-25,25)
        for j in range(100):
            for k in range(100):
                if distance[i, j, k] > 1.5:
                    ax.scatter(-25+j/2,-25+k/2,s=4,c='b')
        plt.savefig(path +"vis/"+str(i)+ ".png",dpi=300)
        plt.close()
    #plt.show()

def plot_spheres(path, frame):


    #distance = np.fromfile(path+"frames/"+str(frame)+"_dists.dat", dtype=np.float32)
    distance  = np.load(path+"frames/"+str(frame)+"_dists.npy")
    #ax = fig.add_subplot(111, projection="3d")
    for i in range(len(distance)):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-25,25)
        ax.set_ylim(-25,25)
        for j in range(100):
            for k in range(100):
                if distance[i, j, k] > 1:
                    ax.scatter(-25+j/2,-25+k/2,s=4,c='b')
        plt.savefig(path +"vis/"+str(i)+ ".png",dpi=300)
        plt.close()
    #plt.show()



def save_dists(path):
    res = pca_coordinates_per_frame(path+"trajectories/trajectory_z.dat")
    min_max = hf.find_global_extrema(res["coords_pca"])
    x_dist = math.ceil(min_max["max_xyz"][0] - min_max["min_xyz"][0])
    x_dist =70
    coords = res["coords_pca"]
    frame = 221
    distance = np.zeros((x_dist*2, 100, 100))
    for i in range(2*x_dist):
        print(i)
        for j in range(100):
            for k in range(100):
                distance[ i, j, k] = hf.findMin(np.array([-42 + i/2, -25+j/2,-25+k/2]),coords[frame])
    np.save(path +"frames/"+str(frame)+"_dists.npy", distance)

def create_graph_handle(path,frame):
    distance  = np.load(path+"frames/"+str(frame)+"_dists.npy")
    graph = ga.Graphical_Analysis(path,frame)
    graph.create_graph(distance)
    return graph





if __name__ == '__main__':

    path = "/home/greg/Documents/oxDNA/NanoCap_Square_V2/"
    save_dists(path)
    #plot_spheres(path,221)
    #graph = create_graph_handle(path,220)
    #print("Created Graph")
    #graph.CreateFastestPathHandle()

    #test = hf.compute_z_density_from_oxdna_dat("/home/greg/Documents/oxDNA/Nano_Cap_Full_V2_Control/trajectories/trajectory_z.dat")

    """
    values = plot_average_profile_after_frame(
        path+"trajectories/trajectory_z.dat",
        start_frame=100,
        bins=120,
        normalize=True,
        title="Average Density",
        savepath=path+"AverageDens_comb_y.png",
    )
    path = "/home/greg/Documents/oxDNA/NanoCap_Full_V3_Control/"
    values = plot_average_profile_after_frame(
        path+"trajectories/trajectory_z.dat",
        start_frame=100,
        bins=120,
        normalize=True,
        ax = values["ax"],
        title="Average Density",
        savepath=path+"AverageDens_comb_y.png",
    )"""
    make_pc1_density_video(path+"trajectories/trajectory_z.dat", out_path=path  + "Density.mp4",
                           bins=120, normalize=True, fps=20)



