import numpy as np
import scipy.interpolate

def interpolate(lr_img, scale_factor):
    h, w, c = lr_img.shape
    h_range = np.array(range(h))
    w_range = np.array(range(w))

    new_h = h*scale_factor
    new_w = w*scale_factor
    hr_img = np.zeros((new_h, new_w, c))
    new_h_range = np.array(range(new_h)) / scale_factor
    new_w_range = np.array(range(new_w)) / scale_factor

    for i in range(c):
        rbs = scipy.interpolate.RectBivariateSpline(h_range, w_range, lr_img[:, :, i])
        hr_img[:, :, i] = rbs(new_h_range, new_w_range)
    
    return hr_img
