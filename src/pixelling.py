import numpy as np
from skimage import io 
from typing import Tuple

def z_tuple(z_ij:complex): 
    return (z_ij.real, z_ij.imag)


class Cell: 
    # INDEX is in reference to its Frame Reference. 
    neighbor_directions = {
        'square'  : ['E',  'S',  'W', 'N'], 
        'hexagon' : ['E', 'SE', 'SW', 'W', 'NW', 'NE'], 
        'triangle': ['SE', 'S', 'SW', 'NW', 'N', 'NE']  # By pairs.  
    }

    def __init__(self, geom, index): 
        self.geom  = geom 
        self.index = index
        self.neighbors  = {}
        self.directions = Cell.neighbor_directions[geom]
        

    # Indices take values (i, j)
    # Complex are the centers at which they stand
    # ... in canonical reference.     
    @classmethod
    def index_2_complex(cls, geom, index) -> complex:
        if geom == 'square': 
            z_ij = complex(index) + 0.5*(1 + 1j)
        
        elif geom == 'hexagon': 
            sin_60 = np.sin(np.pi/3)
            odd_ai = index[0] % 2
            z_ij   = (complex( *(index*np.array((sin_60, 1)))) 
                + 0.5*complex(1, 1+odd_ai))
        return z_ij


    @classmethod
    def complex_2_index(cls, geom, z_ij) -> Tuple[int]: 
        if geom == 'square': 
            pre_index = z_tuple(z_ij - 0.5*(1 + 1j))
        
        elif geom == 'hexagon':
            sin_60 = np.sin(np.pi/3) 
            ai     = np.round((z_ij.real - 0.5)/sin_60)
            odd_ai = ai % 2
            aj     = np.round( z_ij.imag - 0.5*(1+odd_ai))
            pre_index  = (ai, aj)

        return map(int, pre_index)


    def map_neighbor(self, k_ngbr) -> Tuple[int]: 
        if self.geom == 'square': 
            z_root = np.exp(-2*np.pi*1j/4)
            z_add  = 1j*z_root**k_ngbr
        
        elif self.geom == 'hexagon': 
            z_root = np.exp(-2*np.pi*1j/6)
            z_add  = 1j*z_root**k_ngbr

        z_0 = self.coord
        z_1 = z_0 + z_add
        return Cell.complex_2_index(self.geom, z_1)        


    def __repr__(self) -> str: 
        return f"<Cell{self.index} at ({self.coord:.1f})>"



class Frame:
    def __init__(self, size):
        self.size = size


    def set_grid(self, geom, cell_size, **kwargs): 
        self.geom = geom
        self.cell_size = cell_size
        (mm, nn)  = self._set_grid_shape()
        self.grid = [ [Cell(geom, (ii, jj)) for jj in range(nn)] 
                for ii in range(mm) ]  

        for a_cell in self.each_cell():
            z_coord      = Cell.index_2_complex(geom, a_cell.index) 
            a_cell.coord = cell_size*z_coord
            for kk, direction in enumerate(a_cell.directions): 
                i1, j1 = a_cell.map_neighbor(kk)
                a_cell.neighbors[direction] = self[i1, j1]


    def each_cell(self) -> Cell: 
        for cells_row in self.grid:
            for a_cell in cells_row: 
                yield a_cell


    def _set_grid_shape(self) -> Tuple[int]: 
        if self.geom == 'hexagon': 
            csc_60 = 1/np.sin(np.pi/3)        
            mm = np.floor((self.size[0]/self.cell_size - 1)*csc_60) + 1
            nn = np.floor( self.size[1]/self.cell_size )
        else: 
            raise 'Something'   

        self.shape = (int(mm), int(nn))
        return self.shape
 

    def __getitem__(self, ij_tuple) -> Cell: 
        (mm, nn) = self.shape
        (ii, jj) = ij_tuple
        in_range = (ii in range(mm)) and (jj in range(nn)) 
        the_item = self.grid[ii][jj] if in_range else None
        return the_item
    

    def __repr__(self) -> str: 
        if hasattr(self, 'grid'): 
            the_repr = f"<Frame sized {self.size}, w-shape {self.shape}, {self.geom} {self.cell_size}>"
        else: 
            the_repr = f"<Frame sized {self.size}>"
        return the_repr

        

class PictureFrame:
    def __init__(self, file, a_frame=None, **kwargs): 
        self.image = io.imread(file)
        self.shape = self.image.shape[:2]
        
        if a_frame is not None: 
            mount_args = {'flip': 'vert'}
            mount_args.update(kwargs)
            self.mount_frame(a_frame, mount_args)


    def mount_frame(self, a_frame:Frame, flip='vert'): 
        # frame [size, shape, type]
        # let's match size to picture.shape
        self.frame = a_frame 
        self.frame_ratio = np.array(a_frame.size)/self.shape[:2]

        for a_cell in a_frame.each_cell(): 
            a_coord = list(z_tuple(a_cell.coord))
            if flip == 'vert':
                a_coord[0] = a_frame.size[0] - a_coord[0]
                
            a_cell.frame_coord = complex(*a_coord)
        self.frame_coords = np.array(a_cell.frame_coord 
                for a_cell in a_frame.each_cell())


    def set_raster_match(self, **kwargs): 
        if not hasattr(self, 'frame'): 
            raise "Picture doesn't have a frame mounted."
        
        match_args = {'method': 'radius'}
        match_args.update(kwargs)
        
        raster_match = -np.ones(self.shape[:2])

        if match_args['method'] == 'radius': 
            ratio  = self.frame_ratio
            radius = self.frame.cell_size/2

            for kk, a_cell in enumerate(self.frame.each_cell()):
                ci, cj = z_tuple(a_cell.frame_coord)
                uu, vv = map(lambda X: X.flatten().astype(int), 
                    np.meshgrid(
                        np.arange(np.floor((ci - 2*radius)/ratio[0]), np.ceil((ci + 2*radius)/ratio[0])), 
                        np.arange(np.floor((cj - 2*radius)/ratio[1]), np.ceil((cj + 2*radius)/ratio[1])), 
                        indexing='ij'))
                
                points_0 = (ratio[0]*uu + ratio[1]*vv*1j)

                match_0  = (np.isin(uu, np.arange(self.shape[0])) 
                          & np.isin(vv, np.arange(self.shape[1])))
                match_1  = (np.abs(points_0 - a_cell.frame_coord) <= radius) 

                the_match = match_0 & match_1
                raster_match[uu[the_match], vv[the_match]] = kk

        self.raster_match = raster_match    
        return raster_match        


    def set_frame_image(self, **kwargs): 
        if not hasattr(self, 'raster_match'):
            self.match_picture_to_frame(**kwargs)
        
        is_rgb  = self.image.ndim == 3         
        k_scale = kwargs['k_colors'] if 'k_colors' in kwargs else None

        raster_match = self.raster_match
        frame_image  = np.zeros(self.image.shape)
        for kk, a_cell in enumerate(self.frame.each_cell()): 
            k_match = (raster_match == kk)
            a_value = (np.mean(self.image[k_match, :], axis=0)
                if is_rgb else np.mean(self.image[k_match]))
            b_value = (np.round(k_scale*a_value)/k_scale 
                if k_scale else a_value)
            
            frame_image[k_match, :] = a_cell.frame_value = b_value

        the_type = self.image.dtype
        self.frame_image = frame_image.astype(the_type)
    
    
    def print_frame_image(self, **kwargs): 
        if not hasattr(self, 'frame_image'): 
            self.set_frame_image(**kwargs)
        
        io.imshow(self.frame_image)
        io.show()
    



if __name__ == '__main__': 
    
    a_frame = Frame([29, 39])
    a_frame.set_grid('hexagon', 1.2)

    the_file = "data/majo-diego-crop.jpg"
    the_picture = PictureFrame(the_file, a_frame)
    
    the_picture.set_raster_match(method='radius')
    the_picture.print_frame_image()

