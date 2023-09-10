import numpy as np
import copy
import numpy.fft as fft
from scipy.signal import butter, hilbert, freqz
import matplotlib.pyplot as plt
import matplotlib as mlp
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import interpolate as intp


def model_resizing(model0,  bx=None, ex=None, bz=None, ez=None, ssr=(1, 1)):
    """
    model_resizing resizes your model which is dictionary by cutting and interpolation.

    

    Parameters
    ----------
    model0 : dict
        The input model to be resized
    bx : int, optional
        First column of desired model, by default None
    ex : int, optional
        Last column of desired model, by default None
    bz : int, optional
        First row of desired model, by default None
    ez : int, optional
        Last row of desired model, by default None
    ssr : tuple, optional
        Sampling rate for interpolation in Z- and X-directions, by default (1, 1)

    Returns
    -------
    model : dict
        Dictionary containg the resized model
    """
    model = copy.deepcopy(model0)
    for param in model:
        gz, gx = np.mgrid[:model[param].shape[0], :model[param].shape[1]]
        x = np.arange(0, model[param].shape[1], 1)
        z = np.arange(0, model[param].shape[0], 1)
        interpolator = intp.interp2d(x, z, model[param])
        xi = np.arange(0, model[param].shape[1], ssr[1])
        zi = np.arange(0, model[param].shape[0], ssr[0])
        model[param] = interpolator(xi, zi)
        model[param] = model[param].astype(np.float32, order='C')

        model[param] = model[param][bz:ez, bx:ex]
    return model


def prepare_residual(res, s=1.):
    """
    prepare_residual prepares the seismic data as the desire format of FWI class.

    Parameters
    ----------
    res : dict
        Seismic section
    s : ndarray
        Parqameter to create the square matirix of W as the weight of seismic data in cost function.

    Returns
    -------
    dict
        Reformatted seismic section 
    """
    
    data = {}
    shape = res[[*res][0]].shape
    all_comps = ['vx', 'vz', 'taux', 'tauz', 'tauxz']
    
    for param in all_comps:
        if param in res:
            data[param] = s * res[param]
        else:
            data[param] = np.zeros(shape, np.float32)
    return data


def lowpass(x1, highcut, fn, order=1, axis=1, show=False):
    x = copy.deepcopy(x1)

    # Zero padding
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))

    nt = x.shape[axis]

    # Bring the data to frequency domain
    x_fft = fft.fft(x, n=nt, axis=axis)

    # Calculate the highcut btween 0 to 1
    scaled_highcut = 2*highcut/fn

    # Generate the filter
    b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")

    # Get the frequency response
    w, h1 = freqz(b, a, worN=nt, whole=True)
    h = np.diag(h1)

    # Apply the filter in the frequency domain
    fd = h @ x_fft

    #Double filtering by the conjugate to make up the shift
    h = np.diag(np.conjugate(h1))
    fd = h @ fd

    # Bring back to time domaine
    f_inv = fft.ifft(fd, n=nt, axis=axis).real
    f_inv = f_inv[:, :-padding, :]

    return f_inv

def adj_lowpass(x, highcut, fn, order, axis=1):

    # Zero padding
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))

    nt = x.shape[axis]

    # Bring the data to frequency domain
    x_fft = np.fft.fft(x, n=nt, axis=axis)

    # Calculate the highcut btween 0 to 1
    scaled_highcut = 2*highcut / fn

    # Generate the filter
    b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")

    # Get the frequency response
    w, h = freqz(b, a, worN=nt, whole=True)

    # Get the conjugate of the filter
    h_c = np.diag(np.conjugate(h))

    # Apply the adjoint filter in the frequency domain
    fd = h_c @ x_fft

    # Double filtering by the conjugate to make up the shift
    h_c = np.diag(h)
    fd = h_c @ fd

    # Bring back to time domaine
    adj_f_inv = np.fft.ifft(fd, axis=axis).real
    adj_f_inv = adj_f_inv[:, :-padding, :]
    return adj_f_inv


def show_earth_model(model, keys=[],offset=None, depth= None, **kwargs):
    """
    earth_model show the earth model.

    This function is provided to show the earth models.

    Args:
        model (Dictionary): A dictionary containing the earth model.
        keys (list, optional): List of parameters you want to show. Defaults to [].

    Returns:
        fig (class): The figure class  to which the images are added for furthur settings like im.set-clim(). 
    """
    nx = max(model[[*model][0]].shape[1], model[[*model][0]].shape[1])
    nz = max(model[[*model][0]].shape[0], model[[*model][0]].shape[0])
    if offset is None:
        offset = nx
        
    if depth is None:
        depth = nz
        
    if keys == []:
        keys = model.keys()
        
    n = len(keys)
    fig = plt.figure(figsize=(4*n, 4))

    i = 1
    ims = []

    for param in keys:
        ax = fig.add_subplot(1, n, i)
        aspect = (model[param].shape[0]/model[param].shape[1])  

        ax.axis([0, offset, 0, depth])
        ax.set_aspect(aspect)

        im = ax.imshow(model[param], **kwargs)
        ims.append(im)
        axes_divider = make_axes_locatable(ax)
        cax = axes_divider.append_axes('right', size='7%', pad='2%')
        
        fig.colorbar(im, cax=cax, shrink=aspect+0.1,
                        pad=0.01)
        ax.invert_yaxis()
        ax.set_title(param, loc='center')
        if i>1:
            ax.set_yticks([])
        i +=1
    fig.__dict__['ims'] = ims
    
    return fig