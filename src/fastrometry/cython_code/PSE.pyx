from astropy.io import fits
import numpy as np
from libc.math cimport sqrt
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int getCircleSize(int rad):
    cdef:
        int circlepts
        int remainpts
        ### The integers in the following array each represent the number of pixels encased by a circle whose radius is equal to the index plus one.
        ### (see https://en.wikipedia.org/wiki/Gauss_circle_problem)
        np.ndarray[dtype=int,ndim=1] encasedpixelnos = np.array([5,13,29,49,81,113,149,197,253,317,377,441,529,613,709,797,901,1009,1129,1257,1373,1517,1653,1793,1961,2121,2289,2453,2629,2821,3001,3209,3409,3625,3853,4053,4293,4513,4777,5025,5261,5525,5789,6077,6361])

    return encasedpixelnos[rad-1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef makePixelCircle(int rad, int[:,:] pixcircle):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        Py_ssize_t a = 0
        Py_ssize_t b = 0

    for x in range(-rad,rad+1):
        for y in range(-rad,rad+1):
            if x*x+y*y <= rad*rad:
                pixcircle[a,0] = x
                pixcircle[a,1] = y
                a += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray fillBackgroundMap(double[:,:] bgmap_view, double[:,:] img_view, int img_xmax, int img_ymax, int sourcesep, int verbose, int debug):

    cdef:
        Py_ssize_t x
        Py_ssize_t y
        double tmp
        double a
        double b
        double c
        double d
        int leftmost_safe_x = 1 + sourcesep
        int topmost_safe_y = 1 + sourcesep
        int rightmost_safe_x = img_xmax - sourcesep
        int bottommost_safe_y = img_ymax - sourcesep

    for x in range(leftmost_safe_x,rightmost_safe_x+1):
        for y in range(topmost_safe_y,bottommost_safe_y+1):
            a = img_view[(y-sourcesep)-1,(x-sourcesep)-1] #NW
            b = img_view[(y-sourcesep)-1,(x+sourcesep)-1] #NE
            c = img_view[(y+sourcesep)-1,(x+sourcesep)-1] #SE
            d = img_view[(y+sourcesep)-1,(x-sourcesep)-1] #SW
            if a > b:
                tmp = a
                a = b
                b = tmp
            if a > c:
                tmp = a
                a = c
                c = tmp
            if a > d:
                tmp = a
                a = d
                d = tmp
            if b > c:
                tmp = b
                b = c
                c = tmp
            if b > d:
                tmp = b
                b = d
                d = tmp
            if c > d:
                tmp = c
                c = d
                d = tmp
            bgmap_view[y-1,x-1] = b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef findCentroids(int x, int y, Py_ssize_t square_rad, double[:,::] IMGview, double center_background, double[:] centroid_results_view):
    cdef:
        double sumOfWeightsTimesXDists = 0
        double sumOfWeightsTimesYDists = 0
        double sumOfWeights = 0
        Py_ssize_t x_armlength
        Py_ssize_t y_armlength
        double x_centroid
        double y_centroid
    
    if square_rad == 0:
        x_centroid = x
        y_centroid = y
        
    else:
        for x_armlength in range(-square_rad, square_rad+1):
            for y_armlength in range(-square_rad, square_rad+1):
                sumOfWeightsTimesXDists += (IMGview[(y+y_armlength)-1, (x+x_armlength)-1]-center_background)*x_armlength
                sumOfWeightsTimesYDists += (IMGview[(y+y_armlength)-1, (x+x_armlength)-1]-center_background)*y_armlength
                sumOfWeights += (IMGview[(y+y_armlength)-1, (x+x_armlength)-1]-center_background)
        x_centroid = x+sumOfWeightsTimesXDists/sumOfWeights
        y_centroid = y+sumOfWeightsTimesYDists/sumOfWeights

    centroid_results_view[0] = x_centroid
    centroid_results_view[1] = y_centroid
    centroid_results_view[2] = sumOfWeights

@cython.boundscheck(False)
@cython.wraparound(False)
cdef findIslandCentroids(double[:,:] img_view, int[:] islandbnds_view, double[:] cntdresults_view):
    cdef:
        Py_ssize_t x_length
        Py_ssize_t y_length
        double sumOfWeightsTimesXDists = 0
        double sumOfWeightsTimesYDists = 0
        double sumOfWeights = 0
        double x_centroid
        double y_centroid

    if islandbnds_view[1]-islandbnds_view[0] == 0:
        islandbnds_view[0] -= 1     
        islandbnds_view[1] += 1
    if islandbnds_view[3]-islandbnds_view[2] == 0:
        islandbnds_view[2] -= 1
        islandbnds_view[3] += 1
        
    
    for x in range(islandbnds_view[0], islandbnds_view[1]+1):
        for y in range(islandbnds_view[2], islandbnds_view[3]+1):
            sumOfWeightsTimesXDists += img_view[y-1,x-1]*x
            sumOfWeightsTimesYDists += img_view[y-1,x-1]*y
            sumOfWeights += img_view[y-1,x-1]
    x_centroid = sumOfWeightsTimesXDists/sumOfWeights
    y_centroid = sumOfWeightsTimesYDists/sumOfWeights

    cntdresults_view[0] = x_centroid
    cntdresults_view[1] = y_centroid
    cntdresults_view[2] = sumOfWeights

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void doQuickerRecursiveWalk(Py_ssize_t x, Py_ssize_t y, int[:,:] occupymap_view, int[:,:] sindmap_view, int g):
    occupymap_view[y-1,x-1] = g
    sindmap_view[y-1,x-1] = g
    cdef:
        Py_ssize_t surr_x
        Py_ssize_t surr_y
    for surr_x in range(x-1,x+2):
        for surr_y in range(y-1,y+2):
            if occupymap_view[surr_y-1,surr_x-1] == -2:
                doQuickerRecursiveWalk(surr_x,surr_y,occupymap_view,sindmap_view,g)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint isOnIsland(Py_ssize_t surr_x, Py_ssize_t surr_y, double[:,:] img_view, int leftmost_safe_x, int rightmost_safe_x, int topmost_safe_y, int bottommost_safe_y, int[:,:] occupymap_view, double pixsat):
    if leftmost_safe_x <= surr_x <= rightmost_safe_x:
        if topmost_safe_y <= surr_y <= bottommost_safe_y:
            if img_view[surr_y-1,surr_x-1] >= pixsat and occupymap_view[surr_y-1,surr_x-1] == -1:      #if the pixel is in the image, is saturated, and is unclaimed by another source         
                return True
    else:
        return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void doRecursiveWalk(Py_ssize_t x, Py_ssize_t y, double[:,:] img_view, int leftmost_safe_x, int rightmost_safe_x, int topmost_safe_y, int bottommost_safe_y, int[:] islandbnds_view, int[:,:] occupymap_view, double pixsat, int curr_sind):
    occupymap_view[y-1,x-1] = -2
    if x < islandbnds_view[0]:
        islandbnds_view[0] = x
    if x > islandbnds_view[1]:
        islandbnds_view[1] = x
    if y < islandbnds_view[2]:
        islandbnds_view[2] = y
    if y > islandbnds_view[3]:
        islandbnds_view[3] = y
    cdef:
        Py_ssize_t surr_x
        Py_ssize_t surr_y
    for surr_x in range(x-1,x+2):
        for surr_y in range(y-1,y+2):
            if isOnIsland(surr_x,surr_y,img_view,leftmost_safe_x,rightmost_safe_x,topmost_safe_y,bottommost_safe_y,occupymap_view,pixsat):
                doRecursiveWalk(surr_x,surr_y,img_view,leftmost_safe_x,rightmost_safe_x,topmost_safe_y,bottommost_safe_y,islandbnds_view,occupymap_view,pixsat,curr_sind)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int mapSaturationIslands(double[:,:] img_view, int img_xmax, int img_ymax, np.ndarray[dtype=np.double_t, ndim=2] smetadata, int[:,:] occupymap_view, int[:,:] sindmap_initial_view, int[:,:] sindmap_refine_view, int sourcesep, double pixsat, int npts, int nrefinepts, int curr_sind, int verbose, int debug):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        Py_ssize_t g
        int leftmost_safe_x = 1 + sourcesep
        int rightmost_safe_x = img_xmax - sourcesep
        int topmost_safe_y = 1 + sourcesep
        int bottommost_safe_y = img_ymax - sourcesep
        double[:,:] smetadata_view = smetadata
        int[:] islandbnds_view = np.zeros(4,dtype=int)
        double[:] cntdresults_view = np.zeros(3,dtype=np.double)
        int[:,:] newoccupymap_view = -1*np.ones((img_ymax,img_xmax),dtype=np.int32)

    if pixsat > 0:
        for x in range(leftmost_safe_x,rightmost_safe_x+1):
            for y in range(topmost_safe_y,bottommost_safe_y+1):
                if img_view[y-1,x-1] >= pixsat and occupymap_view[y-1,x-1] == -1:
                    islandbnds_view[0] = x
                    islandbnds_view[1] = x
                    islandbnds_view[2] = y
                    islandbnds_view[3] = y
                    doRecursiveWalk(x,y,img_view,leftmost_safe_x,rightmost_safe_x,topmost_safe_y,bottommost_safe_y,islandbnds_view,occupymap_view,pixsat,curr_sind)
                    findIslandCentroids(img_view,islandbnds_view,cntdresults_view)      
                    if curr_sind < nrefinepts:
                        smetadata_view[curr_sind,0] = cntdresults_view[0]           #x centroid
                        smetadata_view[curr_sind,1] = cntdresults_view[1]           #y centroid
                        smetadata_view[curr_sind,2] = cntdresults_view[2]           #total SQUARE kernel sum
                    else:
                        dimmest = smetadata_view[0,2]
                        for p in range(nrefinepts):
                            if smetadata_view[p,2] < dimmest:
                                dimmest = smetadata_view[p,2]
                                dimmest_ind = p       
                        if cntdresults_view[2] > dimmest:
                            smetadata_view[dimmest_ind,0] = cntdresults_view[0]           #x centroid
                            smetadata_view[dimmest_ind,1] = cntdresults_view[1]           #y centroid
                            smetadata_view[dimmest_ind,2] = cntdresults_view[2]           #total SQUARE kernel sum
                    curr_sind += 1

        smetadata_sorted = smetadata[smetadata[:,2].argsort()][::-1]
        smetadata[:] = smetadata_sorted

        if curr_sind < nrefinepts:
            for g in range(curr_sind):
                doQuickerRecursiveWalk(int(smetadata[g,0]), int(smetadata[g,1]), occupymap_view, newoccupymap_view, g)
                occupymap_view[:] = newoccupymap_view[:]
        if curr_sind >= nrefinepts:
            for g in range(nrefinepts):
                if g < npts:
                    doQuickerRecursiveWalk(int(smetadata[g,0]), int(smetadata[g,1]), occupymap_view, sindmap_initial_view, g)
                    doQuickerRecursiveWalk(int(smetadata[g,0]), int(smetadata[g,1]), occupymap_view, sindmap_refine_view, g)
                else:
                    doQuickerRecursiveWalk(int(smetadata[g,0]), int(smetadata[g,1]), occupymap_view, sindmap_refine_view, g)

    if verbose >= 1:
        if curr_sind < nrefinepts:
            print("Found {} saturation islands. Will map remaining sources.".format(curr_sind))
        elif curr_sind >= nrefinepts:
            print("Found {} saturation islands, which is greater than or equal to the required {}. Will cut and stop.".format(curr_sind, nrefinepts))
    
    return curr_sind
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int mapRemainingSources(double[:,:] img_view, int img_xmax, int img_ymax, double img_max, double img_median, np.ndarray[dtype=np.double_t,ndim=2] smetadata, double[:,:] bgmap_view, int[:,:] sindmap_initial_view, int[:,:] sindmap_refine_view, int[:,:] occupymap_view, int kernelrad, int[:,:] kernelcircle_view, int kernelsize, int sourcesep, int[:,:] sourcesepcircle_view, int sourcesepsize, int npts, int nrefinepts, int curr_sind, int verbose, int debug):

    cdef:
        int num_islands = curr_sind     ###The curr_sind reflects the amount of sources found so far, or equivalently the index of the next source
                                        ###to be put in smetadata (since smetadata is 0-based like all numpy arrays). So num_islands is a number
                                        ###representing the next index to be filled after all the saturation islands have been put in smetadata.
        Py_ssize_t iter
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t m
        Py_ssize_t n
        Py_ssize_t p
        Py_ssize_t q
        Py_ssize_t r
        Py_ssize_t s
        Py_ssize_t t
        Py_ssize_t u
        Py_ssize_t x
        Py_ssize_t y
        Py_ssize_t prevx
        Py_ssize_t prevy
        int leftmost_safe_x = 1 + sourcesep
        int topmost_safe_y = 1 + sourcesep
        int rightmost_safe_x = img_xmax - sourcesep
        int bottommost_safe_y = img_ymax - sourcesep
        int div_factor = 8
        double pixelthresh = (img_max-img_median)/div_factor + img_median
        int max_iters = 20
        double centerbg
        double kernelsum
        bint localpeak
        double dimmest
        int num_psesources
        Py_ssize_t dimmest_ind
        double[:,:] smetadata_view = smetadata
        double[:] cntdresults_view = np.zeros(3,dtype=np.double)
    
    sindmap_initial_view[:] = occupymap_view[:]
    sindmap_refine_view[:] = occupymap_view[:]

    for iter in range(max_iters):

        if verbose >= 1:
            print("iteration ",iter)
            print("pixel thresh",pixelthresh)

        for x in range(leftmost_safe_x, rightmost_safe_x+1):
            for y in range(topmost_safe_y, bottommost_safe_y+1):
                if occupymap_view[y-1,x-1] == -1:
                    centerbg = bgmap_view[y-1,x-1]
                    if img_view[y-1,x-1] - centerbg > pixelthresh:
                        localpeak = True        ##in other words, set to True preemptively and test using the surrounding pixels
                        for i in range(sourcesepsize):
                            if img_view[(y+sourcesepcircle_view[i,1])-1,(x+sourcesepcircle_view[i,0])-1] > img_view[y-1,x-1]:
                                localpeak = False
                                break
                        if localpeak == True:
                            for j in range(sourcesepsize):
                                occupymap_view[(y+sourcesepcircle_view[j,1])-1, (x+sourcesepcircle_view[j,0])-1] = -2
                            
                            if curr_sind < nrefinepts-1:                                 
                                ###If the slot about to be filled in smetadata isn't nrefinepts-1 (the nrefinepts'th source), we can now go onto the
                                ###next (x,y) and not perform unhelpful expensive calculations (aka, kernel integration and centroiding). These
                                ###calculations do have to be done at some point, but there's no use doing them now if we end up not finding
                                ###nrefinepts sources and have to do another outermost loop. So we just save the x,y and if this turns out to be
                                ###a "good run", we will come back and do the calculations.

                                smetadata_view[curr_sind,0] = x
                                smetadata_view[curr_sind,1] = y
                                smetadata_view[curr_sind,2] = 0
                                
                                curr_sind += 1

                            elif curr_sind == nrefinepts-1:
                                ###If we got to this point, we know that this loop gives us at least nrefinepts sources. We go back (by accessing
                                ###the smetadata) and perform kernel integration to get the brightnesses of all sources found so far.

                                for k in range(num_islands,nrefinepts-1):
                                    prevx = int(smetadata_view[k,0])
                                    prevy = int(smetadata_view[k,1])
                                    kernelsum = 0
                                    for m in range(kernelsize):
                                        kernelsum += img_view[(prevy+kernelcircle_view[m,1])-1,(prevx+kernelcircle_view[m,0])-1]
                                    kernelsum -= bgmap_view[prevy-1,prevx-1]*kernelsize 
                                    smetadata_view[k,2] = kernelsum
                                
                                ###Now getting the brightness of the current nrefinepts'th source
                                kernelsum = 0
                                for n in range(kernelsize):
                                    kernelsum += img_view[(x+kernelcircle_view[n,1])-1,(y+kernelcircle_view[n,0])-1]
                                kernelsum -= centerbg*kernelsize 

                                smetadata_view[499,0] = x
                                smetadata_view[499,1] = y
                                smetadata_view[499,2] = kernelsum

                                curr_sind += 1
                                
                            else:
                                ###Once we have nrefinepts sources, there are no more unfilled slots in the array. We have to calculate the dimmest source
                                ###in the nrefinepts found and, if this new source is brighter, replace the dimmest one.
                                
                                dimmest = smetadata_view[0,2]
                                for p in range(num_islands,nrefinepts):
                                    if smetadata_view[p,2] < dimmest:
                                        dimmest = smetadata_view[p,2]
                                        dimmest_ind = p
                                        
                                kernelsum = 0
                                for q in range(kernelsize):
                                    kernelsum += img_view[(y+kernelcircle_view[q,1])-1,(x+kernelcircle_view[q,0])-1]
                                kernelsum -= centerbg*kernelsize 
                                if kernelsum > dimmest:
                                    smetadata_view[dimmest_ind,0] = x
                                    smetadata_view[dimmest_ind,1] = y
                                    smetadata_view[dimmest_ind,2] = kernelsum
                                
                                curr_sind += 1


        if curr_sind < nrefinepts:      ###from this point it is easiest to think of curr_sind representing
                                        ###the amount of sources found after iterating through all pixels
            div_factor *= 2
            pixelthresh = (img_max-img_median)/div_factor + img_median
        else:
            break

    smetadata_sorted = smetadata[smetadata[:,2].argsort()][::-1]   ###sort smetadata by brightness
    smetadata[:] = smetadata_sorted

    if curr_sind < nrefinepts:
        ###If max_iters iterations have been performed and we still don't have nrefinepts sources,
        ###we just have to take what we have and do all the heavy calculations for those now.

        num_psesources = curr_sind

        for r in range(num_islands,num_psesources):
            prevx = int(smetadata_view[r,0])
            prevy = int(smetadata_view[r,1])
            kernelsum = 0
            if r < npts:
                for s in range(kernelsize):
                    kernelsum += img_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1]
                    sindmap_refine_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1] = r
                    sindmap_initial_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1] = r
            else:
                for s in range(kernelsize):
                    kernelsum += img_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1]
                    sindmap_refine_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1] = r
            kernelsum -= bgmap_view[prevy-1,prevx-1]*kernelsize 

            findCentroids(prevx,prevy,kernelrad,img_view,bgmap_view[prevy-1,prevx-1],cntdresults_view)
            smetadata_view[r,0] = cntdresults_view[0]   #x centroid
            smetadata_view[r,1] = cntdresults_view[1]   #y centroid
            smetadata_view[r,2] = kernelsum

    else:
        ###If we have nrefinepts or more sources, we have already calculated kernel sums for
        ###the purpose of replacing the dimmest sources, so the final steps involve populating
        ###the kernels in the sindmap and finding centroids

        num_psesources = nrefinepts

        for t in range(num_islands,nrefinepts):
            prevx = int(smetadata_view[t,0])
            prevy = int(smetadata_view[t,1])
            if t < npts:
                for u in range(kernelsize):
                    sindmap_refine_view[(prevy+kernelcircle_view[u,1])-1,(prevx+kernelcircle_view[u,0])-1] = t
                    sindmap_initial_view[(prevy+kernelcircle_view[u,1])-1,(prevx+kernelcircle_view[u,0])-1] = t
            else:
                for u in range(kernelsize):
                    sindmap_refine_view[(prevy+kernelcircle_view[u,1])-1,(prevx+kernelcircle_view[u,0])-1] = t
            findCentroids(prevx,prevy,kernelrad,img_view,bgmap_view[prevy-1,prevx-1],cntdresults_view)
            smetadata_view[t,0] = cntdresults_view[0]   #x centroid
            smetadata_view[t,1] = cntdresults_view[1]   #y centroid

    if verbose >= 1:
        if curr_sind < nrefinepts:
            print("Unsuccessful at finding {} sources. Found {} sources".format(nrefinepts,num_psesources))
        elif curr_sind >= nrefinepts:
            print("Found {} sources, which is greater than or equal to the required minimum of {}.".format(curr_sind, nrefinepts))

    return num_psesources

def PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, sindmap_initial, sindmap_refine, smetadata, verbose, debug):

    if debug >= 1:
        import matplotlib.pyplot as plt

    bgmap = np.zeros((img_ymax,img_xmax),dtype=np.double)
    fillBackgroundMap(bgmap, img, img_xmax, img_ymax, sourcesep, verbose, debug)

    curr_sind = 0
    occupymap = -1*np.ones(img.shape,dtype=np.int32)
    curr_sind = mapSaturationIslands(img, img_xmax, img_ymax, smetadata, occupymap, sindmap_initial, sindmap_refine, sourcesep, pixsat, npts, nrefinepts, curr_sind, verbose, debug)     #return curr_sind to use in mapRemainingSources

    if verbose >= 1:
        print("Found {} saturation islands.".format(curr_sind))

    if debug >= 1:
        plt.imshow(occupymap)
        plt.colorbar()
        plt.show()
        plt.imshow(sindmap_initial)
        plt.colorbar()
        plt.show()
        plt.imshow(sindmap_refine)
        plt.colorbar()
        plt.show()

    if curr_sind >= nrefinepts:
        num_psesources = nrefinepts
        return num_psesources

    kernelsize = getCircleSize(kernelrad)
    kernelcircle = np.zeros((kernelsize,2),dtype=np.int32)
    makePixelCircle(kernelrad,kernelcircle)

    sourcesepsize = getCircleSize(sourcesep)
    sourcesepcircle = np.zeros((sourcesepsize,2),dtype=np.int32)
    makePixelCircle(sourcesep,sourcesepcircle)

    img_median = np.median(img)
    img_max = np.max(img)

    num_psesources = mapRemainingSources(img, img_xmax, img_ymax, img_max, img_median, smetadata, bgmap, sindmap_initial, sindmap_refine, occupymap, kernelrad, kernelcircle, kernelsize, sourcesep, sourcesepcircle, sourcesepsize, npts, nrefinepts, curr_sind, verbose, debug)

    if debug >= 1:
        plt.imshow(occupymap)
        plt.colorbar()
        plt.show()
        plt.imshow(sindmap_initial)
        plt.colorbar()
        plt.show()
        plt.imshow(sindmap_refine)
        plt.colorbar()
        plt.show()

    return num_psesources

                        
    


