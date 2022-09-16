import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color as clr



def get_frame_grid(a_shape, diam, indexing='ij', grid_type='hex'): 
    if grid_type == 'hex': 
        csc_60 = 1/np.sin(np.pi/3)

        b_shape = a_shape if indexing == 'ij' else reversed(a_shape)    
        
        mm = int(np.floor((b_shape[0]/diam - 1)*csc_60)) + 1
        nn = int(np.floor( b_shape[1]/diam))

        hex_shape = (mm, nn) if indexing == 'ij' else (nn, mm)
        return hex_shape


def stack_hex_centers(h_shape, indexing='ij'): 
    # arrange unit circles starting at (1/2, 1/2) moving horizontal, 
    # then stacking vertical-wise. 

    sin_60 = np.sin(np.pi/3)

    h2_shape = h_shape if indexing == 'ij' else reversed(h_shape)
    u_i = np.arange(h2_shape[0])*sin_60 + 1/2
    v_j = np.arange(h2_shape[1]) + 1/2

    odd_ones = np.mod(np.arange(h2_shape[0]), 2)[:, None]

    u_mat, v_mat = np.meshgrid(u_i, v_j, indexing=indexing)
    uu =  u_mat.flatten()
    vv = (v_mat + odd_ones/2).flatten()

    uv_centers = np.column_stack((uu, vv))    
    return uv_centers
    

def label_grid(grid_values, centers, radius, pxl_type='hex'): 

    grid_labels = -np.ones(grid_values).astype(int)
    
    if isinstance(radius, np.ndarray): 
        radius = np.average(radius)

    # Some considerations for when PXL_TYPE == 'HEX'
    angles_6 = np.exp((np.arange(6)+1/2)*np.pi*1j/3)
    rays_6   = 2*radius*np.append(0+0j, angles_6)

    for cc in range(centers.shape[0]): 
        cx, cy = centers[cc, :]  

        uu, vv = map(lambda X: X.flatten().astype(int), 
            np.meshgrid(
                np.arange(np.floor(cx - 2*radius), np.ceil(cx + 2*radius)), 
                np.arange(np.floor(cy - 2*radius), np.ceil(cy + 2*radius)), 
                indexing='ij'))

        # Complexify, and reversed on purpose. 
        grid_points = (uu + vv*1j)  
        a_center    = (cx + cy*1j)

        pre_assign  = (uu < grid_values[0]) & (vv < grid_values[1])

        if  pxl_type == 'hex': 
            ctr_rivals   = a_center + rays_6
            complex_dist = np.abs(grid_points[:, None] - ctr_rivals)
            post_assign  = (np.argmin(complex_dist, 1) == 0) 
            # Closer to a_center, than othern center rivals. 

        elif pxl_type == 'circle': 
            post_assign = (np.abs(grid_points - (cx+cy*1j)) <= radius)
            # Just within the circle of given RADIUS.  

        pxl_assign = pre_assign & post_assign
        grid_labels[uu[pxl_assign], vv[pxl_assign]] = cc        

    row_rev = list(reversed(range(grid_values[0]))) 
    col_rev = list(reversed(range(grid_values[1])))
    pre_grid = grid_labels[row_rev, :]
    return pre_grid


def average_by_label(grid_coords, grid_labels): 

    max_label  = np.max(grid_labels)
    new_coords = np.zeros(grid_coords.shape)
    new_values = np.zeros(max_label)

    for k in range(max_label):
        which_k = (grid_labels == k)
        
        new_values[k] = np.average(grid_coords[which_k])  
        new_coords[which_k] = new_values[k]

    return new_coords, new_values




if __name__ == '__main__': 
    a_picture = io.imread("./data/majo-268-360.jpg")
    gray_pic  = clr.rgb2gray(a_picture)

    box_shape = np.array([39, 29])  # cms:  width, length. 

    pxl_ratio = a_picture.shape[:2]/box_shape

    cork_dmtr =  2.4 #2.4 cms 

    hx_shape  = get_frame_grid(box_shape, cork_dmtr, 'ij', 'hex')

    # Big Pixels Grid 
    PXL_centers = cork_dmtr*np.matmul(
        stack_hex_centers(hx_shape), np.diag(pxl_ratio))
    
    pxl_2_PXL_circ = label_grid(
        gray_pic.shape, PXL_centers, cork_dmtr/2*pxl_ratio, 'circle')

    pxl_2_PXL_hex = label_grid(
        gray_pic.shape, PXL_centers, cork_dmtr/2*pxl_ratio, 'hex')

    GRAY_pic_circ, values_cir = average_by_label(gray_pic, pxl_2_PXL_circ)

    GRAY_pic_hex, values_hex  = average_by_label(gray_pic, pxl_2_PXL_hex)

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
    






