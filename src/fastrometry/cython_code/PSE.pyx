from astropy.io import fits
import numpy as np
from libc.math cimport sqrt
cimport numpy as np
import cython

def debuggerPlot(array, user_dir, debug_report, savename, figsize=(7,7), bottom=0.25, vmin=None, vmax=None, colbar=False, colbar_extend=False, title=None, dscrp=None, bmargin=0.1, disable_console=False, collist=None, boundslist=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.colors import ListedColormap
    if colbar_extend is True:
        extend = 'both'
    else:
        extend = 'neither'
    if disable_console is True:
        import sys, os
        console = sys.stderr    
        sys.stderr = open(os.devnull,'w')
    if collist is not None:
        cm = ListedColormap(collist)
    else:
        cm = None
    if boundslist is not None:
        nm = BoundaryNorm(boundslist,ncolors=2)
    else:
        nm = None
    plt.figure(figsize=figsize)
    plt.subplots_adjust(bottom=bottom)
    plt.imshow(array, cmap=cm, norm=nm, vmin=vmin, vmax=vmax)
    if colbar is True:
        plt.colorbar(extend=extend)
    plt.title(title)
    plt.figtext(0.5, bmargin, dscrp, ha="center", fontsize=9)
    plt.savefig(user_dir+"\\debug\\"+debug_report+"\\{}".format(savename))
    plt.show()
    if disable_console is True:
        sys.stderr = console

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void createROI(int[:,:] roi_view, int img_xmax, int img_ymax, int pixelradius, str shape):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        double img_xcenter = img_xmax/2
        double img_ycenter = img_ymax/2
        double img_xrad2 = (img_xcenter)*(img_xcenter)
        double img_yrad2 = (img_ycenter)*(img_ycenter)
        double pixelradius2 = pixelradius*pixelradius

    if shape == "circle":
        for x in range(1,img_xmax+1):
            for y in range(1,img_ymax+1):
                if (x-img_xcenter)*(x-img_xcenter)/pixelradius2+(y-img_ycenter)*(y-img_ycenter)/pixelradius2 > 1:
                    roi_view[y-1,x-1] = 0
    return

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
cdef np.ndarray fillBackgroundMap(double[:,:] bgmap_view, double[:,:] img_view, int img_xmax, int img_ymax, int sourcesep):

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
cdef void doQuickerRecursiveWalk(Py_ssize_t x, Py_ssize_t y, int[:,:] occupymap_view, int[:,:] srcindexmap_view, int g):
    occupymap_view[y-1,x-1] = g
    srcindexmap_view[y-1,x-1] = g
    cdef:
        Py_ssize_t surr_x
        Py_ssize_t surr_y
    for surr_x in range(x-1,x+2):
        for surr_y in range(y-1,y+2):
            if occupymap_view[surr_y-1,surr_x-1] == -2:
                doQuickerRecursiveWalk(surr_x,surr_y,occupymap_view,srcindexmap_view,g)

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
cdef int mapSaturationIslands(double[:,:] img_view, int img_xmax, int img_ymax, int[:,:] roi_view, np.ndarray[dtype=np.double_t, ndim=2] pse_metadata, int[:,:] occupymap_view, int[:,:] srcindexmap_initial_view, int[:,:] srcindexmap_refine_view, int sourcesep, double pixsat, int npts, int nrefinepts, int curr_sind, str user_dir, str debug_report, int verbosity, int debug):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        Py_ssize_t g
        int leftmost_safe_x = 1 + sourcesep
        int rightmost_safe_x = img_xmax - sourcesep
        int topmost_safe_y = 1 + sourcesep
        int bottommost_safe_y = img_ymax - sourcesep
        double[:,:] pse_metadata_view = pse_metadata
        int[:] islandbnds_view = np.zeros(4,dtype=int)
        double[:] cntdresults_view = np.zeros(3,dtype=np.double)
        int[:,:] newoccupymap_view = -1*np.ones((img_ymax,img_xmax),dtype=np.int32)

    if pixsat > 0:
        for x in range(leftmost_safe_x,rightmost_safe_x+1):
            for y in range(topmost_safe_y,bottommost_safe_y+1):
                if roi_view[y-1,x-1] == 1:
                    if img_view[y-1,x-1] >= pixsat and occupymap_view[y-1,x-1] == -1:
                        islandbnds_view[0] = x
                        islandbnds_view[1] = x
                        islandbnds_view[2] = y
                        islandbnds_view[3] = y
                        doRecursiveWalk(x,y,img_view,leftmost_safe_x,rightmost_safe_x,topmost_safe_y,bottommost_safe_y,islandbnds_view,occupymap_view,pixsat,curr_sind)
                        findIslandCentroids(img_view,islandbnds_view,cntdresults_view)      
                        if curr_sind < nrefinepts:
                            pse_metadata_view[curr_sind,0] = cntdresults_view[0]           #x centroid
                            pse_metadata_view[curr_sind,1] = cntdresults_view[1]           #y centroid
                            pse_metadata_view[curr_sind,2] = cntdresults_view[2]           #total SQUARE kernel sum
                        else:
                            dimmest = pse_metadata_view[0,2]
                            for p in range(nrefinepts):
                                if pse_metadata_view[p,2] < dimmest:
                                    dimmest = pse_metadata_view[p,2]
                                    dimmest_ind = p       
                            if cntdresults_view[2] > dimmest:
                                pse_metadata_view[dimmest_ind,0] = cntdresults_view[0]           #x centroid
                                pse_metadata_view[dimmest_ind,1] = cntdresults_view[1]           #y centroid
                                pse_metadata_view[dimmest_ind,2] = cntdresults_view[2]           #total SQUARE kernel sum
                        curr_sind += 1

        pse_metadata_sorted = pse_metadata[pse_metadata[:,2].argsort()][::-1]
        if debug:
            np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_islandsonly.csv",pse_metadata,delimiter=",")
            np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_islandsonly_sorted.csv",pse_metadata_sorted,delimiter=",")
        pse_metadata[:] = pse_metadata_sorted

        if curr_sind < nrefinepts:
            for g in range(curr_sind):
                doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), occupymap_view, newoccupymap_view, g)
                occupymap_view[:] = newoccupymap_view[:]
        if curr_sind >= nrefinepts:
            for g in range(nrefinepts):
                if g < npts:
                    doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), occupymap_view, srcindexmap_initial_view, g)
                    doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), occupymap_view, srcindexmap_refine_view, g)
                else:
                    doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), occupymap_view, srcindexmap_refine_view, g)

    return curr_sind
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int mapRemainingSources(double[:,:] img_view, int img_xmax, int img_ymax, double img_max, double img_median, int[:,:] roi_view, np.ndarray[dtype=np.double_t,ndim=2] pse_metadata, double[:,:] bgmap_view, int[:,:] srcindexmap_initial_view, int[:,:] srcindexmap_refine_view, int[:,:] occupymap_view, int kernelrad, int[:,:] kernelcircle_view, int kernelsize, int sourcesep, int[:,:] sourcesepcircle_view, int sourcesepsize, int npts, int nrefinepts, int curr_sind, str user_dir, str debug_report, int verbosity, int debug):

    cdef:
        int num_islands = curr_sind     ###The curr_sind reflects the amount of sources found so far, or equivalently the index of the next source
                                        ###to be put in pse_metadata (since pse_metadata is 0-based like all numpy arrays). So num_islands is a number
                                        ###representing the next index to be filled after all the saturation islands have been put in pse_metadata.
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
        Py_ssize_t r2
        Py_ssize_t s2
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
        double[:,:] pse_metadata_view = pse_metadata
        double[:] cntdresults_view = np.zeros(3,dtype=np.double)
    
    srcindexmap_initial_view[:] = occupymap_view[:]
    srcindexmap_refine_view[:] = occupymap_view[:]

    if verbosity == 2:
        print("| | Trying various pixel thresholds...")

    for iter in range(max_iters):
        for x in range(leftmost_safe_x, rightmost_safe_x+1):
            for y in range(topmost_safe_y, bottommost_safe_y+1):
                if roi_view[y-1,x-1] == 1:
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
                                    ###If the slot about to be filled in pse_metadata isn't nrefinepts-1 (the nrefinepts'th source), we can now go onto the
                                    ###next (x,y) and not perform unhelpful expensive calculations (aka, kernel integration and centroiding). These
                                    ###calculations do have to be done at some point, but there's no use doing them now if we end up not finding
                                    ###nrefinepts sources and have to do another outermost loop. So we just save the x,y and if this turns out to be
                                    ###a "good run", we will come back and do the calculations.

                                    pse_metadata_view[curr_sind,0] = x
                                    pse_metadata_view[curr_sind,1] = y
                                    pse_metadata_view[curr_sind,2] = 0
                                    
                                    curr_sind += 1

                                elif curr_sind == nrefinepts-1:
                                    ###If we got to this point, we know that this loop gives us at least nrefinepts sources. We go back (by accessing
                                    ###the pse_metadata) and perform kernel integration to get the brightnesses of all sources found so far.

                                    for k in range(num_islands,nrefinepts-1):
                                        prevx = int(pse_metadata_view[k,0])
                                        prevy = int(pse_metadata_view[k,1])
                                        kernelsum = 0
                                        for m in range(kernelsize):
                                            kernelsum += img_view[(prevy+kernelcircle_view[m,1])-1,(prevx+kernelcircle_view[m,0])-1]
                                        kernelsum -= bgmap_view[prevy-1,prevx-1]*kernelsize 
                                        pse_metadata_view[k,2] = kernelsum
                                    
                                    ###Now getting the brightness of the current nrefinepts'th source
                                    kernelsum = 0
                                    for n in range(kernelsize):
                                        kernelsum += img_view[(x+kernelcircle_view[n,1])-1,(y+kernelcircle_view[n,0])-1]
                                    kernelsum -= centerbg*kernelsize 

                                    pse_metadata_view[nrefinepts-1,0] = x
                                    pse_metadata_view[nrefinepts-1,1] = y
                                    pse_metadata_view[nrefinepts-1,2] = kernelsum

                                    curr_sind += 1
                                    
                                else:
                                    ###Once we have nrefinepts sources, there are no more unfilled slots in the array. We have to calculate the dimmest source
                                    ###in the nrefinepts found and, if this new source is brighter, replace the dimmest one.
                                    
                                    dimmest = np.inf
                                    for p in range(num_islands,nrefinepts):
                                        if pse_metadata_view[p,2] < dimmest:
                                            dimmest = pse_metadata_view[p,2]
                                            dimmest_ind = p
                                            
                                    kernelsum = 0
                                    for q in range(kernelsize):
                                        kernelsum += img_view[(y+kernelcircle_view[q,1])-1,(x+kernelcircle_view[q,0])-1]
                                    kernelsum -= centerbg*kernelsize 
                                    if kernelsum > dimmest:
                                        pse_metadata_view[dimmest_ind,0] = x
                                        pse_metadata_view[dimmest_ind,1] = y
                                        pse_metadata_view[dimmest_ind,2] = kernelsum


                                    curr_sind += 1

        if verbosity == 2:
            print("| | | Iteration: {}".format(iter))
            print("| | | Pixel threshold: {}".format(pixelthresh))
            print("| | | Sources found: {}".format(curr_sind))

        if curr_sind < nrefinepts:  ###Here think of curr_sind representing the amount of sources found this iteration
            div_factor *= 2
            pixelthresh = (img_max-img_median)/div_factor + img_median
        else:
            break

    if debug:
        np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_allsources.csv",pse_metadata,delimiter=",")

    if curr_sind < nrefinepts:
        ###If max_iters iterations have been performed and we still don't have nrefinepts sources,
        ###we just have to take what we have and do all the heavy calculations for those now.

        if verbosity == 2:
            print("| | done")

        for r in range(num_islands,curr_sind):
            prevx = int(pse_metadata_view[r,0])
            prevy = int(pse_metadata_view[r,1])
            kernelsum = 0
            for s in range(kernelsize):
                kernelsum += img_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1]
            kernelsum -= bgmap_view[prevy-1,prevx-1]*kernelsize
            pse_metadata_view[r,2] = kernelsum

        pse_metadata_sorted = pse_metadata[pse_metadata[:,2].argsort()][::-1]   ###sort pse_metadata by brightness
        if debug:
            np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_allsources_sorted.csv",pse_metadata_sorted,delimiter=",")
        num_zeros = nrefinepts - curr_sind      ###If not all nrefinepts slots are filled, there will be leftover zeros that need to be pushed to the back of the array
        for rr in range(num_islands, nrefinepts):
            if pse_metadata_sorted[rr,2] < 0:
                pse_metadata_sorted[rr-num_zeros,:] = pse_metadata_sorted[rr,:]
                pse_metadata_sorted[rr,:] = 0
        if debug:
            np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_allsources_sorted_zerosmoved.csv",pse_metadata_sorted,delimiter=",")
        pse_metadata[:] = pse_metadata_sorted

        for r2 in range(num_islands, curr_sind):
            prevx = int(pse_metadata_view[r2,0])
            prevy = int(pse_metadata_view[r2,1])
            if r2 < npts:
                for s2 in range(kernelsize):
                    srcindexmap_refine_view[(prevy+kernelcircle_view[s2,1])-1,(prevx+kernelcircle_view[s2,0])-1] = r2
                    srcindexmap_initial_view[(prevy+kernelcircle_view[s2,1])-1,(prevx+kernelcircle_view[s2,0])-1] = r2
            else:
                for s2 in range(kernelsize):
                    srcindexmap_refine_view[(prevy+kernelcircle_view[s2,1])-1,(prevx+kernelcircle_view[s2,0])-1] = r2

            findCentroids(prevx,prevy,kernelrad,img_view,bgmap_view[prevy-1,prevx-1],cntdresults_view)
            pse_metadata_view[r2,0] = cntdresults_view[0]   #x centroid
            pse_metadata_view[r2,1] = cntdresults_view[1]   #y centroid

    elif curr_sind >= nrefinepts:

        if verbosity == 2:
            print("| | done")

        pse_metadata_sorted = pse_metadata[pse_metadata[:,2].argsort()][::-1]   ###sort pse_metadata by brightness
        if debug:
            np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_allsources_sorted.csv",pse_metadata_sorted,delimiter=",")
        pse_metadata[:] = pse_metadata_sorted

        ###If we have nrefinepts or more sources, we have already calculated kernel sums for
        ###the purpose of replacing the dimmest sources, so the final steps involve populating
        ###the kernels in the srcindexmap and finding centroids

        for t in range(num_islands,nrefinepts):
            prevx = int(pse_metadata_view[t,0])
            prevy = int(pse_metadata_view[t,1])
            if t < npts:
                for u in range(kernelsize):
                    srcindexmap_refine_view[(prevy+kernelcircle_view[u,1])-1,(prevx+kernelcircle_view[u,0])-1] = t
                    srcindexmap_initial_view[(prevy+kernelcircle_view[u,1])-1,(prevx+kernelcircle_view[u,0])-1] = t
            else:
                for u in range(kernelsize):
                    srcindexmap_refine_view[(prevy+kernelcircle_view[u,1])-1,(prevx+kernelcircle_view[u,0])-1] = t
            findCentroids(prevx,prevy,kernelrad,img_view,bgmap_view[prevy-1,prevx-1],cntdresults_view)
            pse_metadata_view[t,0] = cntdresults_view[0]   #x centroid
            pse_metadata_view[t,1] = cntdresults_view[1]   #y centroid

    return curr_sind

def PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, pixelradius, shape, srcindexmap_initial, srcindexmap_refine, pse_metadata, user_dir, debug_report, filename, verbosity, debug):

    num_psesources = 0
    skipRemainingSources = False
    
    if verbosity == 2:
        print("| Creating ROI map...")
    roi = np.ones((img_ymax,img_xmax),dtype=np.int)
    createROI(roi, img_xmax, img_ymax, pixelradius, shape)
    if debug:
        description = "Region to be scanned by PSE. Specified by the options shape and buffer."
        debuggerPlot(roi, user_dir, debug_report, "ROI.png", title="ROI", dscrp=description)
    if verbosity == 2:
        print("| done")

    if verbosity == 2:
        print("| Creating background map...")
    bgmap = np.zeros((img_ymax,img_xmax),dtype=np.double)
    fillBackgroundMap(bgmap, img, img_xmax, img_ymax, sourcesep)
    if debug:
        description = "Used for determining the amplitude-above-background for each pixel. \nThe PSE does not scan pixels in a thin ({}-pixel-wide) strip around \nthe border, so the background is not calculated there.".format(sourcesep)
        debuggerPlot(bgmap, user_dir, debug_report, "bgmap.png", bottom=0.2, vmin=np.percentile(bgmap,1), vmax=np.percentile(bgmap,99), colbar=True, colbar_extend=True, title="Background map", dscrp=description)
    if verbosity == 2:
        print("| done")

    curr_sind = 0
    occupymap = -1*np.ones(img.shape,dtype=np.int)

    if verbosity >= 1:
        print("| Finding saturation islands...")
    curr_sind = mapSaturationIslands(img, img_xmax, img_ymax, roi, pse_metadata, occupymap, srcindexmap_initial, srcindexmap_refine, sourcesep, pixsat, npts, nrefinepts, curr_sind, user_dir, debug_report, verbosity, debug)     #return curr_sind to use in mapRemainingSources
    if curr_sind < nrefinepts:
        if verbosity >= 1:
            print("| | Found {} saturation islands.".format(curr_sind))
            print("| done")
    elif curr_sind >= nrefinepts:
        if verbosity >= 1:
            print("| | Found {} saturation islands. Trimmed to the {} brightest.".format(curr_sind,nrefinepts))
            if debug:
                description = "The occupied source kernels for the top {} brightest sources \nin the image. Used to find an initial WCS solution.".format(npts)
                debuggerPlot(srcindexmap_initial, user_dir, debug_report, "srcindexmap_initial.png", bottom=0.2, title="srcindexmap_initial", dscrp=description, bmargin=0.07, disable_console=True, collist=["blue","yellow"], boundslist=[-1,0,npts])
                description = "The occupied source kernels for the top {} brightest sources \nin the image. Used to refine the WCS solution.".format(nrefinepts)
                debuggerPlot(srcindexmap_refine, user_dir, debug_report, "srcindexmap_refine.png", bottom=0.2, title="srcindexmap_refine", dscrp=description, bmargin=0.07, disable_console=True, collist=["blue","yellow"], boundslist=[-1,0,nrefinepts])
            print("| done")
        num_psesources = nrefinepts
        skipRemainingSources = True

    if skipRemainingSources is False:
        kernelsize = getCircleSize(kernelrad)
        kernelcircle = np.zeros((kernelsize,2),dtype=np.int32)
        makePixelCircle(kernelrad,kernelcircle)

        sourcesepsize = getCircleSize(sourcesep)
        sourcesepcircle = np.zeros((sourcesepsize,2),dtype=np.int32)
        makePixelCircle(sourcesep,sourcesepcircle)

        img_median = np.median(img)
        img_max = np.max(img)

        if verbosity == 2:
            print("| Median amplitude: {}".format(img_median))
            print("| Max amplitude: {}".format(img_max))

        if verbosity >= 1:
            print("| Finding remaining sources...")
        curr_sind = mapRemainingSources(img, img_xmax, img_ymax, img_max, img_median, roi, pse_metadata, bgmap, srcindexmap_initial, srcindexmap_refine, occupymap, kernelrad, kernelcircle, kernelsize, sourcesep, sourcesepcircle, sourcesepsize, npts, nrefinepts, curr_sind, user_dir, debug_report, verbosity, debug)
        if curr_sind < nrefinepts:
            if verbosity >= 1:
                print("| | Unsuccessful at finding {} sources. Found {} sources.".format(nrefinepts, curr_sind))
                if debug:
                    description = "The occupied source kernels for the top {} brightest sources \nin the image. Used to find an initial WCS solution.".format(npts)
                    debuggerPlot(srcindexmap_initial, user_dir, debug_report, "srcindexmap_initial.png", bottom=0.2, title="srcindexmap_initial", dscrp=description, bmargin=0.07, disable_console=True, collist=["blue","yellow"], boundslist=[-1,0,npts])
                    description = "The occupied source kernels for the top {} brightest sources \nin the image. Used to refine the WCS solution.".format(nrefinepts)
                    debuggerPlot(srcindexmap_refine, user_dir, debug_report, "srcindexmap_refine.png", bottom=0.2, title="srcindexmap_refine", dscrp=description, bmargin=0.07, disable_console=True, collist=["blue","yellow"], boundslist=[-1,0,nrefinepts])
                print("| done")
            num_psesources = curr_sind
        elif curr_sind >= nrefinepts:
            if verbosity >= 1:
                print("| | Found {} sources total. Trimmed to the {} brightest.".format(curr_sind, nrefinepts))
                if debug:
                    description = "The occupied source kernels for the top {} brightest sources \nin the image. Used to find an initial WCS solution.".format(npts)
                    debuggerPlot(srcindexmap_initial, user_dir, debug_report, "srcindexmap_initial.png", bottom=0.2, title="srcindexmap_initial", dscrp=description, bmargin=0.07, disable_console=True, collist=["blue","yellow"], boundslist=[-1,0,npts])
                    description = "The occupied source kernels for the top {} brightest sources \nin the image. Used to refine the initial WCS solution.".format(nrefinepts)
                    debuggerPlot(srcindexmap_refine, user_dir, debug_report, "srcindexmap_refine.png", bottom=0.2, title="srcindexmap_refine", dscrp=description, bmargin=0.07, disable_console=True, collist=["blue","yellow"], boundslist=[-1,0,nrefinepts])
                print("| done")
            num_psesources = nrefinepts

    if debug:
        np.savetxt(user_dir+"\\debug\\"+debug_report+"\\pse_metadata.csv",pse_metadata,delimiter=",")

    if debug:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits
        plt.figure(figsize=(10,8))
        image_data = fits.getdata('{}\\{}'.format(user_dir,filename))
        plt.subplots_adjust(bottom=0.15)
        plt.imshow(image_data, cmap="gray", norm=LogNorm())
        plt.scatter(pse_metadata[:,0],pse_metadata[:,1],color='red',marker='.')
        plt.title("pse_metadata centroids")
        dscrp="Centroids from the pse_metadata, plotted overtop the image."
        plt.figtext(0.5, 0.05, dscrp, ha="center", fontsize=9)
        plt.savefig(user_dir+"\\debug\\"+debug_report+"\\pse_metadata_centroids.png")
        plt.show()

    return num_psesources

                        
    


