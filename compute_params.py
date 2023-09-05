import argparse
import os
import subprocess
import sys

def install(packages):
    """Install a list of packages.
    """
    for p in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    import tifffile as tiff
    from skimage import measure
    from skimage import io
    from scipy.spatial import Delaunay
    import numpy as np
    import pandas as pd

except ImportError as e:
    print("[Warning] Some packages are missing. Installing...")
    install(['tifffile', 'scikit-image', 'scipy', 'numpy', 'pandas'])
    import tifffile as tiff
    from skimage import measure
    from skimage import io
    from scipy.spatial import Delaunay
    import numpy as np
    import pandas as pd

#---------------------------------------------------------------------------
# Image reader

def tif_read_meta(tif_path, display=False):
    """
    read the metadata of a tif file and stores them in a python dict.
    if there is a 'ImageDescription' tag, it transforms it as a dictionary
    """
    meta = {}
    with tiff.TiffFile(tif_path) as tif:
        for page in tif.pages:
            for tag in page.tags:
                tag_name, tag_value = tag.name, tag.value
                if display: print(tag.name, tag.code, tag.dtype, tag.count, tag.value)

                # below; fix storage problem for ImageDescription tag
                if tag_name == 'ImageDescription': 
                    list_desc = tag_value.split('\n')
                    dict_desc = {}
                    for idx, elm in enumerate(list_desc):
                        split = elm.split('=')
                        dict_desc[split[0]] = split[1]
                    meta[tag_name] = dict_desc
                else:
                    meta[tag_name] = tag_value
            break # just check the first image
    return meta

def tif_get_spacing(path, res=1e-6):
    """
    get the image spacing stored in the metadata file.
    """
    img_meta = tif_read_meta(path)

    xres = (img_meta["XResolution"][1]/img_meta["XResolution"][0])*res
    yres = (img_meta["YResolution"][1]/img_meta["YResolution"][0])*res
    zres = float(img_meta["ImageDescription"]["spacing"])*res

    return (xres, yres, zres)

def tif_read_imagej(img_path):
    """Read tif file metadata stored in a ImageJ format.
    adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8

    Parameters
    ----------
    img_path : str
        Path to the input image.

    Returns
    -------
    img : numpy.ndarray
        Image.
    img_meta : dict
        Image metadata. 
    """

    with tiff.TiffFile(img_path) as tif:
        assert tif.is_imagej

        # store img_meta
        img_meta = {}

        # get image resolution from TIFF tags
        tags = tif.pages[0].tags
        x_resolution = tags['XResolution'].value
        y_resolution = tags['YResolution'].value
        resolution_unit = tags['ResolutionUnit'].value
        
        img_meta["resolution"] = (x_resolution, y_resolution, resolution_unit)

        # parse ImageJ metadata from the ImageDescription tag
        ij_description = tags['ImageDescription'].value
        ij_description_metadata = tiff.tifffile.imagej_description_metadata(ij_description)
        # remove conflicting entries from the ImageJ metadata
        ij_description_metadata = {k: v for k, v in ij_description_metadata.items()
                                   if k not in 'ImageJ images channels slices frames'}

        img_meta["description"] = ij_description_metadata
        
        # compute spacing
        xres = (x_resolution[1]/x_resolution[0])
        yres = (y_resolution[1]/y_resolution[0])
        zres = float(ij_description_metadata["spacing"])
        
        img_meta["spacing"] = (xres, yres, zres)

        # read the whole image stack and get the axes order
        series = tif.series[0]
        img = series.asarray()
        
        img_meta["axes"] = series.axes
    
    return img, img_meta

def imread(img_path):
    """
    use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .nii.gz --> SimpleITK.imread
    """
    extension = img_path[img_path.rfind('.'):].lower()
    if extension == ".tif" or extension == ".tiff":
        try: 
            img, img_meta = tif_read_imagej(img_path)  # try loading ImageJ metadata for tif files
            return img, img_meta
        except:   
            img_meta = {}    
            try: img_meta["spacing"] = tif_get_spacing(img_path)
            except: img_meta["spacing"] = []
    
            return io.imread(img_path), img_meta 
    else:
        print("[Error] Unknown image format:", extension)

#---------------------------------------------------------------------------
# Utils for volume meshing

def mesh3d(verts, img, remove_label=0):
    """Creates a 3D mesh from a list of vertices. Uses the image as a mask to remove unwanted triangles.
    Returns list of triangles.
    https://forum.image.sc/t/create-3d-volume-mesh/34052/9
    """
    # Delaunay
    tri = Delaunay(verts).simplices

    # filter inner triangles only
    center = verts[tri[:,0]] + verts[tri[:,1]] + verts[tri[:,2]] + verts[tri[:,3]]
    msk = img[tuple((center/4).astype(int).T)] != remove_label
    tri_sorted = tri[msk]
    return tri_sorted

def tetramesh_vol(a, b, c, d):
    """Volume of a tetrahedron mesh. a, b, c, d are the tetrahedron vertices with 3d coodinates.
    shape of a, b, c, or d are either (3,) or (N,3) where N is the number of vertices in the mesh.
    https://en.wikipedia.org/wiki/Tetrahedron#General_properties
    """
    denom = ((a-d)*(np.cross((b-d),(c-d))))
    denom = denom.sum(axis = 1 if len(denom.shape) > 1 else 0)
    return abs(denom).sum()/6

def compute_sphericity(volume, surface):
    """Return sphericity.
    https://en.wikipedia.org/wiki/Sphericity
    """
    return (pow(np.pi,1/3)*pow(6*volume,2/3))/surface

#---------------------------------------------------------------------------
# Compute volume, surface and mesh

def safe_imread(img_path, spacing=()):
    # read image
    img, metadata = imread(img_path)
    if len(metadata['spacing'])==3 and len(spacing)==0: spacing = np.array(metadata['spacing'])

    assert len(img.shape)==3 or (len(img.shape)==4 and img.shape[0]==1), "[Error] Strange image shape ({}). Please provide a 3d image".format(img.shape)
    
    # sanity check: only 0 or 1 label are allowed
    unq, counts = np.unique(img, return_counts=True)
    if len(unq)!=2:
        print("[Warning] Only 2 class annotations are allowed (0 or 1) but found {}. A threshold will be applied but might causes some issues.".format(unq))
        # set background voxels to 0 and foreground to 1
        img = (img != unq[np.argmax(counts)]).astype(np.uint8)

    # warning if no spacing
    if len(spacing)==0:
        print("[Warning] No spacing has been defined. The result will be expressed in voxel units.")
    
    return img, spacing

def compute_volume_surface_sphericity(img, bg=None, spacing=(), verbose=False):
    """Compute volume, surface and sphericity of a volumetric object.

    Parameters
    ----------
    img : numpy.ndarray
        Image array.
    bg : int, default=None
        Value of the background voxels. If bg is None then use the most frequent value.
    spacing : tuple, default=()
        Image spacing.
    verbose : boolean, default=False
        Whether to display information.
    """
    if bg is None:
        # compute volume with voxel
        labels = measure.label(img)
        unq,vol_voxel = np.unique(labels, return_counts=True)

        if len(unq)>2:
            print("[Warning] More than one object were found in the image. Number of connected components: {}".format(len(unq)-1))

        if verbose: print("Voxel volume:", vol_voxel[1:])

        # use the biggest volume as background
        bg = unq[np.argmax(vol_voxel)]
    
    # Marching cube
    verts, faces, normals, values = measure.marching_cubes(img, 0.5) 
    
    # create and sort correct tetrahedron to obtain a volume mesh
    tetra = mesh3d(verts=verts, img=img, remove_label=bg)
    
    # compute the volume
    volume = tetramesh_vol(verts[tetra[:,0]],verts[tetra[:,1]],verts[tetra[:,2]],verts[tetra[:,3]])
    
    # compute the surface
    surface = measure.mesh_surface_area(verts=verts, faces=faces)
    
    # compute the sphericity
    sphericity = compute_sphericity(volume=volume, surface=surface)
    
    # eventually adapt to spacing
    if len(spacing)>0:
        volume = volume*np.prod(spacing)
        surface = surface*np.prod(spacing)
    
    # display for debugging
    if verbose:
        labels = measure.label(img, background=bg)
        unq,vol_voxel = np.unique(labels, return_counts=True)
        vol_voxel = np.sum(vol_voxel[1:])
        if len(spacing)>0: vol_voxel=vol_voxel*np.prod(spacing)
        print("Number of labels:", unq)
        print("Volume (voxel):", vol_voxel)
        print("Volume (mesh):", volume)
        print("Surface:", surface)
        print("Sphericity:", sphericity)
    
    return volume, surface, sphericity

def compute_flatness_elongation(img, bg=None, spacing=(), verbose=False):
    """Compute flatness and elongation of a volumetric object.
    These are called "Shape factor": https://en.wikipedia.org/wiki/Shape_factor_%28image_analysis_and_microscopy%29#Elongation_shape_factor

    Parameters
    ----------
    img : numpy.ndarray
        Image array.
    bg : int, default=None
        Value of the background voxels. If bg is None then use the most frequent value.
    spacing : tuple, default=()
        Image spacing.
    verbose : boolean, default=False
        Whether to display information.
    """
    if bg is None:
        # compute volume with voxel
        labels = measure.label(img)
        unq,vol_voxel = np.unique(labels, return_counts=True)

        if len(unq)>2:
            print("[Warning] More than one object were found in the image. Number of connected components: {}".format(len(unq)-1))

        if verbose: print("Voxel volume:", vol_voxel[1:])

        # use the biggest volume as background
        bg = unq[np.argmax(vol_voxel)]

    # get foreground voxel coordinates
    fg = np.argwhere(img != bg)

    # compute barycenter and center the foreground voxels
    bary = np.mean(fg,axis=0)
    fg = fg - bary
    # fg = fg / np.sqrt(np.mean(fg*fg))

    # get the covariance matrix
    cov = fg.T.dot(fg)/len(fg)

    # other method:
    # m = measure.moments_central(img, order=2)
    # cov = np.array([
    #     [m[2,0,0], m[1,1,0], m[1,0,1]],
    #     [m[1,1,0], m[0,2,0], m[0,1,1]],
    #     [m[1,0,1], m[0,1,1], m[0,0,2]],
    # ])/m[0,0,0]

    # eventually resize image
    if len(spacing)>0:
        spacing = np.array(spacing).reshape(1,3)
        cov = cov*(spacing.T.dot(spacing))

    # get the eigenvalues
    eigval = np.real(sorted(np.linalg.eig(cov)[0]))

    # compute flatness and elongation
    flatness = np.sqrt(eigval[1]/eigval[0])
    elongation = np.sqrt(eigval[2]/eigval[1])

    return flatness, elongation

def compute_number_vmean_vtot(img, cc_img, bg=None, spacing=(), verbose=False):
    """
    """
    if bg is None:
        # compute volume with voxel
        labels = measure.label(img)
        unq,vol_voxel = np.unique(labels, return_counts=True)

        if len(unq)>2:
            print("[Warning] More than one object were found in the image. Number of connected components: {}".format(len(unq)-1))

        if verbose: print("Voxel volume:", vol_voxel[1:])

        # use the biggest volume as background
        bg = unq[np.argmax(vol_voxel)]
    
    # select only chromocenters in the nucleus
    # CAREFUL: the connectivity is set to 1!
    labels = measure.label(np.logical_and(img!=bg,cc_img!=bg).astype(int), background=bg, connectivity=1)
    
    # compute volumes
    unq,vol_voxel = np.unique(labels, return_counts=True)

    # number_vmean_vtot
    number = len(unq)-1
    vmean = np.mean(vol_voxel[unq!=bg])
    vtot = np.sum(vol_voxel[unq!=bg])

    # No need because images are already resampled!
    # set the spacing if needed
    if len(spacing)>0:
        vmean = vmean*np.prod(spacing)
        vtot = vtot*np.prod(spacing)

    return number, vmean, vtot

    
class ComputeParams:
    """Compute Nucleus and Chromocenter parameters.

    Parameters
    ----------
    nc_path : str
        Path to the nucleus image.
    bg : int, default=0
        Value of the background voxels.
    spacing : tuple, default=()
        Image spacing.
    verbose : boolean, default=False
        Whether to display information.
    """
    NUCLEUS_KEYS = [
        'volume',
        'surface',
        'sphericity',
        'flatness',
        'elongation',
    ]

    CHROMOCENTER_KEYS = [
        'cc_number',
        'cc_vmean',
        'cc_vtot',
        'RHF',
    ]

    def __init__(self, nc_path, cc_path=None, bg=0, spacing=(), verbose=False):
        # stores nucleus and chromocenter parameters
        self.nc_params = {}
        self.cc_params = {}

        # path and spacing
        self.nc_path = nc_path
        self.spacing  = np.array(spacing, dtype=np.float64)

        # information
        if verbose: print("Background voxel is set to", bg)

        # read nucleus image and metadata
        self.nc_imag, self.spacing = safe_imread(img_path=self.nc_path, spacing=self.spacing)

        # read chromocenter image and metadata
        if cc_path is not None:
            self.cc_imag, _ = safe_imread(img_path=cc_path, spacing=self.spacing)

        # nucleus volume, surface and sphericity computation
        self.nc_params['volume'], self.nc_params['surface'], self.nc_params['sphericity'] = compute_volume_surface_sphericity(self.nc_imag, bg=bg, spacing=self.spacing, verbose=verbose)

        # nucleus flatness and elongation
        self.nc_params['flatness'], self.nc_params['elongation'] = compute_flatness_elongation(self.nc_imag, bg=bg, spacing=self.spacing, verbose=verbose)

        # chromocenters computation
        if cc_path is not None:
            self.cc_params['cc_number'], self.cc_params['cc_vmean'], self.cc_params['cc_vtot'] = compute_number_vmean_vtot(img=self.nc_imag, cc_img=self.cc_imag, bg=bg, spacing=self.spacing, verbose=verbose)

            # add RHF
            self.cc_params['RHF'] = self.cc_params['cc_vtot']/self.nc_params['volume']
        else:
            self.cc_params['cc_number'], self.cc_params['cc_vmean'], self.cc_params['cc_vtot'] = 0, 0, 0
            self.cc_params['RHF'] = np.inf

        
    
    def __str__(self):
        out = "filename: {}\n".format(self.nc_path)
        out += "".join("{}: {}\n".format(k, v) for k,v in self.nc_params.items()) # nucleus
        out += "".join("{}: {}\n".format(k, v) for k,v in self.cc_params.items()) # chromocenter
        return out


def compute_directory(path, cc_path=None, bg=0, spacing=(), out_path="params.csv", verbose=False):
    """Same as compute_volume_surface_sphericity but on a directory. Output results in a csv file.
    """
    if len(spacing)>0:
        spacing = np.array(spacing, dtype=np.float64)

    filenames = os.listdir(path)
    out_params = {'filename': filenames}
    for k in ComputeParams.NUCLEUS_KEYS: out_params[k]=[]
    for k in ComputeParams.CHROMOCENTER_KEYS: out_params[k]=[]

    for i in range(len(filenames)):
        print("[{}/{}] Computing parameters for {}".format(i,len(filenames),filenames[i]))
        img_path = os.path.join(path, filenames[i])
        if cc_path is not None: 
            cc_img_path = os.path.join(cc_path, filenames[i])
            if not os.path.exists(cc_img_path): cc_img_path = None
        else: cc_img_path = None

         # compute parameters for that image
        comp_params = ComputeParams(nc_path=img_path, cc_path=cc_img_path, bg=bg, spacing=spacing, verbose=verbose)

        # store nucleus parameters in the output dictionary
        for k,v in comp_params.nc_params.items():
            out_params[k] += [v]
        
        # store chromocenter parameters
        for k,v in comp_params.cc_params.items():
            out_params[k] += [v]

    df = pd.DataFrame(out_params)
    df.to_csv(out_path, index=False)
    # df.to_excel("params.xlsx", index=False)

#---------------------------------------------------------------------------
# Argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Compute objects characteristics.")
    parser.add_argument("-p", "--path", type=str, 
        help="Path to an image or a folder of images.")
    parser.add_argument("-cc", "--chromo", type=str, default=None,
        help="Path to a chromocenter image or a folder of chromocenter images.")
    parser.add_argument("-s", "--spacing", type=str, nargs='+', default=(),
        help="Image spacing. Example: 0.1032 0.1032 0.2")
    parser.add_argument("-b", "--bg_value", type=int, default=0,
        help="(default=0) Value of the background voxels.")
    parser.add_argument("-v", "--verbose", default=False,  action='store_true', dest='verbose',
        help="Display some information.") 
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        compute_directory(path=args.path, cc_path=args.chromo, bg=args.bg_value, spacing=args.spacing, out_path="params.csv", verbose=args.verbose)
    else:
        params = ComputeParams(nc_path=args.path, cc_path=args.chromo, spacing=args.spacing, verbose=args.verbose)
        print(params)
