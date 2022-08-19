
from re import I
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color as clr



def hex_size(xy_shape, diam): 
    csc_60 = 1/np.sin(np.pi/3)

    m_x = np.floor(xy_shape[0]/diam)
    n_y = np.floor(xy_shape[1]/diam*csc_60 - (csc_60-1))

    return m_x, n_y


def hex_centers_sp(mn_shape): 
    # arrange unit circles starting at (1/2, 1/2) moving horizontal, 
    # then stacking vertical-wise. 

    sin_60 = np.sin(np.pi/3)

    u_i = np.arange(mn_shape[0]) + 1/2
    v_j = np.arange(mn_shape[1])*sin_60 + 1/2

    odd_ones = np.mod(np.arange(mn_shape[1]), 2)

    uu, vv = np.meshgrid(u_i, v_j, indexing='ij')
    xx = (uu + odd_ones/2).flatten()
    yy =  vv.flatten()

    xy_centers = np.column_stack((xx, yy))    
    return xy_centers
    

def pixels_2_PIXELS(a_shape, centers, radius, pxl_type='hex'): 
    # X=(i, j) is assigned to C=(CX, CY) for which DIST(X, C) $lt R. 

    # Fill in this MAP_MATRIX with the closest corresponding row number. 
    # Non-assigned are kept as -1  (0 is reserved for 0-row).
    pix_map  = -np.ones(a_shape)
    
    if isinstance(radius, np.ndarray): 
        radius = np.average(radius)

    # Some considerations for when PXL_TYPE == 'HEX'
    hex_radius = radius/np.sin(np.pi/3)
    y_radius = hex_radius if pxl_type == 'hex' else radius
    angles_6 = np.exp((np.arange(6))*np.pi*1j/3)
    rays_6   = 2*radius*np.append(0+0j, angles_6)

    for cc in range(centers.shape[0]): 
        cx, cy = centers[cc, :]
        xx, yy = map(lambda X: X.flatten(), 
            np.meshgrid(
                np.arange(np.floor(cx - radius),   np.ceil(cx + radius)), 
                np.arange(np.floor(cy - y_radius), np.ceil(cy + y_radius)), 
                indexing='ij'))
        # Complexify
        grid_points = (xx + yy*1j)
        a_center    = (cx + cy*1j)

        pre_assign  = (xx < a_shape[0]) & (yy < a_shape[1])

        if  pxl_type == 'hex': 
            ctr_rivals   = a_center + rays_6
            complex_dist = np.abs(grid_points[:, None] - ctr_rivals)
            post_assign  = (np.argmin(complex_dist, 1) == 0) 
            # Closer to a_center, than othern center rivals. 

        elif pxl_type == 'circle': 
            post_assign = (np.abs(grid_points - (cx+cy*1j)) <= radius)
            # Just within the circle of given RADIUS.  

        pxl_assign = pre_assign & post_assign
        pix_map[xx[pxl_assign].astype(int), yy[pxl_assign].astype(int)] = cc        

    return pix_map


def gray_2_GRAY(gray_pic, pxl_2_PXL): 

    GRAY_pic = np.zeros(gray_pic.shape)
    
    for k in range(int(np.max(pxl_2_PXL))):
        which_k = (pxl_2_PXL == k)
        GRAY_pic[which_k] = np.average(gray_pic[which_k]) 

    return GRAY_pic



if __name__ == '__main__': 
    a_picture = io.imread("./data/majo-diego-crop.jpg")
    gray_pic  = clr.rgb2gray(a_picture)

    box_shape = np.array([29, 39])  # cms:  width, length. 

    pxl_ratio = a_picture.shape[:2]/box_shape

    cork_dmtr = 1.2 #2.4 cms 

    mn_shape  = hex_size(box_shape, cork_dmtr)

    # Big Pixels Grid 
    PXL_centers = cork_dmtr*np.matmul(
        hex_centers_sp(mn_shape), np.diag(pxl_ratio))
    
    pxl_2_PXL_circ = pixels_2_PIXELS(
        gray_pic.shape, PXL_centers, cork_dmtr/2*pxl_ratio, 'circle')

    pxl_2_PXL_hex = pixels_2_PIXELS(
        gray_pic.shape, PXL_centers, cork_dmtr/2*pxl_ratio, 'hex')

    GRAY_pic_circ = gray_2_GRAY(gray_pic, pxl_2_PXL_circ)

    GRAY_pic_hex  = gray_2_GRAY(gray_pic, pxl_2_PXL_hex)

    fig, axes = plt.subplots(2, 2)
    ax = axes.ravel()
    ax[0].imshow(a_picture)
    ax[0].set_title('Original')
    ax[1].imshow(gray_pic, cmap=plt.cm.gray)
    ax[1].set_title('Gray Scale')
    ax[2].imshow(GRAY_pic_circ, cmap=plt.cm.gray)
    ax[2].set_title('Pixelated-Circle')
    ax[3].imshow(GRAY_pic_hex, cmap=plt.cm.gray)
    ax[3].set_title('Pixelated-Hex')

    fig.tight_layout()
    plt.show()
    






