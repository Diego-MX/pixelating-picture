# The pictures follow this process: 
# 0. Get a picture. 
# 1. Input options: (out-size, Pixel-shape {sq | hex}, colors {rbg, bw, (c1,c2)}
#    (How do we count out-size Pixels in hex-shaped lattice?)
# 2. Interface to fine tune inputs: 
#    a) Select rectangle from picture in given ratio. 
#    b) Set output pixels:  {Continuous | Discrete (w. range)}.
# 3. Calculations: 
#    i)   Find Pixel centers. 
#    ii)  Get Pixel averages. 
#    iii) Set mapping RGB -> Output. 
#    iv)  Draw Pixeled outcome. 
# 4. Output: 
#    a) The new picture. 
#    b) The keys to set the colors. 
#    c) Bonus if printable PDF. 

# Second Part. 
# If the Pixels are taken from corks.  Take a picture and find which cork goes 
# with each Pixel, via identification methods. 