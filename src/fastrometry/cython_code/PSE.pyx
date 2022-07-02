from astropy.io import fits
import numpy as np
from libc.math cimport sqrt
cimport numpy as np
import cython

def printEvent(f):
    """
    A decorator function that causes messages to be printed to the print both before and after
    a process (for a total of two lines) if the verbosity is at the required level.
    The purpose of a decorator is to return a wrapper, which itself supplements the original
    function with additional commands.
    """
    def wrapper(*args, printconsole=None, **kwargs):
        startmessage = printconsole[0]
        endmessage = printconsole[1]
        verbosity = printconsole[2]
        levelneeded = printconsole[3]
        if verbosity >= levelneeded:
            print(startmessage)
        fout = f(*args,**kwargs)
        if verbosity >= levelneeded:
            print(endmessage)
        return fout
    return wrapper

def debuggerPlot(array, debug_report, savename, figsize=(7,7), bottom=0.2, vmin=None, vmax=None, colbar=False, colbar_extend=False, title=None, dscrp=None, textpos=0.1, colors=None, boundaries=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.colors import ListedColormap
 
    if colbar_extend is True:
        extend = 'both'
    else:
        extend = 'neither'

    if colors is None:
        cm = None
    else:
        cm = ListedColormap(colors)

    if boundaries is None:
        nm = None
    else:
        nm = Normalize(vmin=boundaries[0],vmax=boundaries[1]+1)

    plt.figure(figsize=figsize)
    plt.subplots_adjust(bottom=bottom)
    plt.imshow(array, cmap=cm, norm=nm, vmin=vmin, vmax=vmax)
    if colbar is True:
        plt.colorbar(extend=extend)
    plt.title(title)
    plt.figtext(0.5, textpos, dscrp, ha="center", fontsize=9)
    plt.savefig(debug_report/savename)
    plt.show()

cdef createROI(int[:,:] roi_view, int img_xmax, int img_ymax, int pixelradius, str shape):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        int img_xcenter = img_xmax/2
        int img_ycenter = img_ymax/2
        int pixelradius2 = pixelradius*pixelradius

    if shape == "circle":
        for x in range(1,img_xmax+1):
            for y in range(1,img_ymax+1):
                if (x-img_xcenter)*(x-img_xcenter)+(y-img_ycenter)*(y-img_ycenter) > pixelradius2:
                    roi_view[y-1,x-1] = 0

cdef fillBackgroundMap(double[:,:] bgmap_view, double[:,:] img_view, int img_xmax, int img_ymax, int leftmostsafe_x, int rightmostsafe_x, int topmostsafe_y, int bottommostsafe_y, int sourcesep):

    cdef:
        Py_ssize_t x
        Py_ssize_t y
        double tmp
        double a
        double b
        double c
        double d

    for x in range(leftmostsafe_x,rightmostsafe_x+1):
        for y in range(topmostsafe_y,bottommostsafe_y+1):
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


cdef findCentroids(int x, int y, int square_rad, double[:,:] img_view, double centerbg, double[:] cntdresults_view):
    """
    Uses a square kernel. Return array also includes square kernel sum.
    """

    cdef:
        Py_ssize_t x_armlength
        Py_ssize_t y_armlength
        double sumOfWeightsTimesXDists = 0
        double sumOfWeightsTimesYDists = 0
        double sumOfWeights = 0
        double x_centroid
        double y_centroid
    
    if square_rad == 0:
        x_centroid = x
        y_centroid = y
        
    else:
        for x_armlength in range(-square_rad, square_rad+1):
            for y_armlength in range(-square_rad, square_rad+1):
                sumOfWeightsTimesXDists += (img_view[(y+y_armlength)-1, (x+x_armlength)-1]-centerbg)*x_armlength
                sumOfWeightsTimesYDists += (img_view[(y+y_armlength)-1, (x+x_armlength)-1]-centerbg)*y_armlength
                sumOfWeights += (img_view[(y+y_armlength)-1, (x+x_armlength)-1]-centerbg)
        x_centroid = x+sumOfWeightsTimesXDists/sumOfWeights
        y_centroid = y+sumOfWeightsTimesYDists/sumOfWeights

    cntdresults_view[0] = x_centroid
    cntdresults_view[1] = y_centroid
    cntdresults_view[2] = sumOfWeights


cdef findIslandCentroids(double[:,:] img_view, int[:] islandbnds_view, double[:] cntdresults_view):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        double sumOfWeightsTimesXDists = 0
        double sumOfWeightsTimesYDists = 0
        double sumOfWeights = 0
        double x_centroid
        double y_centroid

    #for single-pixel islands, expand the island to a 3X3
    if islandbnds_view[1] == islandbnds_view[0]:
        islandbnds_view[0] -= 1     
        islandbnds_view[1] += 1
    if islandbnds_view[3] == islandbnds_view[2]:
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


cdef doQuickerRecursiveWalk(Py_ssize_t x, Py_ssize_t y, int g, int[:,:] occupymap_view, int[:,:] srcindexmap_view):
    cdef:
        Py_ssize_t surr_x
        Py_ssize_t surr_y
    
    occupymap_view[y-1, x-1] = g
    srcindexmap_view[y-1, x-1] = g

    for surr_x in range(x-1, x+2):
        for surr_y in range(y-1, y+2):
            if occupymap_view[surr_y-1, surr_x-1] == -2:
                doQuickerRecursiveWalk(surr_x, surr_y, g, occupymap_view, srcindexmap_view)


cdef bint isOnIsland(Py_ssize_t surr_x, Py_ssize_t surr_y, double[:,:] img_view, int leftmostsafe_x, int rightmostsafe_x, int topmostsafe_y, int bottommostsafe_y, double pixsat, int[:,:] occupymap_view):
    """
    A pixel is on the island if it is saturated, unclaimed by another source, and in the image
    """
    if img_view[surr_y-1, surr_x-1] >= pixsat:
        if occupymap_view[surr_y-1, surr_x-1] == -1:
            if leftmostsafe_x <= surr_x <= rightmostsafe_x and topmostsafe_y <= surr_y <= bottommostsafe_y:
                return True
    else:
        return False


cdef doRecursiveWalk(Py_ssize_t x, Py_ssize_t y, double[:,:] img_view, int leftmostsafe_x, int rightmostsafe_x, int topmostsafe_y, int bottommostsafe_y, double pixsat, int[:] islandbnds_view, int[:,:] occupymap_view, int curr_src_ind):
    cdef:
        Py_ssize_t surr_x
        Py_ssize_t surr_y
    
    occupymap_view[y-1,x-1] = -2    #-2 indicates the presence of an island

    if x < islandbnds_view[0]:
        islandbnds_view[0] = x
    if x > islandbnds_view[1]:
        islandbnds_view[1] = x
    if y < islandbnds_view[2]:
        islandbnds_view[2] = y
    if y > islandbnds_view[3]:
        islandbnds_view[3] = y

    for surr_x in range(x-1, x+2):
        for surr_y in range(y-1, y+2):
            if isOnIsland(surr_x, surr_y, img_view, leftmostsafe_x, rightmostsafe_x, topmostsafe_y, bottommostsafe_y, pixsat, occupymap_view):
                doRecursiveWalk(surr_x, surr_y, img_view, leftmostsafe_x, rightmostsafe_x, topmostsafe_y, bottommostsafe_y, pixsat, islandbnds_view, occupymap_view, curr_src_ind)

cdef int mapSaturationIslands(double[:,:] img_view, int img_xmax, int img_ymax, int leftmostsafe_x, int rightmostsafe_x, int topmostsafe_y, int bottommostsafe_y, int[:,:] roi_view, np.ndarray[ndim=2,dtype=np.double_t] pse_metadata, int[:,:] occupymap_view, int[:,:] srcindexmap_initial_view, int[:,:] srcindexmap_refine_view, int sourcesep, double pixsat, int npts, int nrefinepts, int curr_src_ind, debug_report, int verbosity, bint debug):
    cdef:
        Py_ssize_t x
        Py_ssize_t y
        Py_ssize_t p
        Py_ssize_t g

        double[:,:] pse_metadata_view = pse_metadata
        int[:] islandbnds_view = np.zeros(4,dtype=int)
        double[:] cntdresults_view = np.zeros(3,dtype=np.double)

    if pixsat == 0:         #no saturation in image
        return 0        #curr_src_ind, since no islands can be found
    else:
        for x in range(leftmostsafe_x,rightmostsafe_x+1):
            for y in range(topmostsafe_y,bottommostsafe_y+1):
                if img_view[y-1,x-1] >= pixsat:
                    if roi_view[y-1,x-1] == 1:
                        if occupymap_view[y-1,x-1] == -1: 
                            islandbnds_view[0] = x
                            islandbnds_view[1] = x
                            islandbnds_view[2] = y
                            islandbnds_view[3] = y
                            doRecursiveWalk(x, y, img_view, leftmostsafe_x, rightmostsafe_x, topmostsafe_y, bottommostsafe_y, pixsat, islandbnds_view, occupymap_view, curr_src_ind)
                            findIslandCentroids(img_view, islandbnds_view, cntdresults_view)      
                            if curr_src_ind < nrefinepts:
                                pse_metadata_view[curr_src_ind,0] = cntdresults_view[0]    #x centroid
                                pse_metadata_view[curr_src_ind,1] = cntdresults_view[1]    #y centroid
                                pse_metadata_view[curr_src_ind,2] = cntdresults_view[2]    #rectangular kernel sum
                            else:
                                dimmest = np.inf
                                for p in range(nrefinepts):
                                    if pse_metadata_view[p,2] < dimmest:
                                        dimmest = pse_metadata_view[p,2]
                                        dimmest_ind = p       
                                if cntdresults_view[2] > dimmest:
                                    pse_metadata_view[dimmest_ind,0] = cntdresults_view[0]  #x centroid
                                    pse_metadata_view[dimmest_ind,1] = cntdresults_view[1]  #y centroid
                                    pse_metadata_view[dimmest_ind,2] = cntdresults_view[2]  #rectangular kernel sum
                            curr_src_ind += 1

        if debug:
            np.savetxt(debug_report/"pse_metadata_islandsonly.csv", pse_metadata, delimiter=",")

        pse_metadata_sorted = pse_metadata[pse_metadata[:,2].argsort()][::-1]
        pse_metadata[:] = pse_metadata_sorted
        
        if debug:
            np.savetxt(debug_report/"pse_metadata_islandsonly_sorted.csv", pse_metadata, delimiter=",")

        if curr_src_ind < nrefinepts:
            for g in range(curr_src_ind):
                doQuickerRecursiveWalk(int(pse_metadata_view[g,0]), int(pse_metadata_view[g,1]), g, occupymap_view, occupymap_view)     #the second occupymap_view is just to satisfy the function arguments
        if curr_src_ind >= nrefinepts:
            for g in range(nrefinepts):
                if g < npts:
                    doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), g, occupymap_view, srcindexmap_initial_view)
                    doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), g, occupymap_view, srcindexmap_refine_view)
                else:
                    doQuickerRecursiveWalk(int(pse_metadata[g,0]), int(pse_metadata[g,1]), g, occupymap_view, srcindexmap_refine_view)

        return curr_src_ind

cdef int getCircleSize(int rad):
    cdef:
        int circlepts
        int remainpts
        ### The integers in the following array each represent the number of pixels encased by a circle whose radius is equal to the index plus one.
        ### (see https://en.wikipedia.org/wiki/Gauss_circle_problem)
        np.ndarray[dtype=int,ndim=1] encasedpixelnos = np.array([5,13,29,49,81,113,149,197,253,317,377,441,529,613,709,797,901,1009,1129,1257,1373,1517,1653,1793,1961,2121,2289,2453,2629,2821,3001,3209,3409,3625,3853,4053,4293,4513,4777,5025,5261,5525,5789,6077,6361])

    return encasedpixelnos[rad-1]

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


cdef int mapRemainingSources(double[:,:] img_view, int img_xmax, int img_ymax, double img_max, double img_median, int leftmostsafe_x, int rightmostsafe_x, int topmostsafe_y, int bottommostsafe_y, int[:,:] roi_view, np.ndarray[ndim=2,dtype=np.double_t] pse_metadata, double[:,:] bgmap_view, int[:,:] srcindexmap_initial_view, int[:,:] srcindexmap_refine_view, int[:,:] occupymap_view, int kernelrad, int[:,:] kernelcircle_view, int kernelsize, int sourcesep, int[:,:] sourcesepcircle_view, int sourcesepsize, int npts, int nrefinepts, int curr_src_ind, debug_report, int verbosity, bint debug):

    cdef:
        int num_islands = curr_src_ind     ###The curr_src_ind reflects the amount of sources found so far, or equivalently the index of the next source
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
        for x in range(leftmostsafe_x, rightmostsafe_x+1):
            for y in range(topmostsafe_y, bottommostsafe_y+1):
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
                                
                                if curr_src_ind < nrefinepts-1:                                 
                                    ###If the slot about to be filled in pse_metadata isn't nrefinepts-1 (the nrefinepts'th source), we can now go onto the
                                    ###next (x,y) and not perform unhelpful expensive calculations (aka, kernel integration and centroiding). These
                                    ###calculations do have to be done at some point, but there's no use doing them now if we end up not finding
                                    ###nrefinepts sources and have to do another outermost loop. So we just save the x,y and if this turns out to be
                                    ###a "good run", we will come back and do the calculations.

                                    pse_metadata_view[curr_src_ind,0] = x
                                    pse_metadata_view[curr_src_ind,1] = y
                                    pse_metadata_view[curr_src_ind,2] = 0
                                    
                                    curr_src_ind += 1

                                elif curr_src_ind == nrefinepts-1:
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

                                    curr_src_ind += 1
                                    
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


                                    curr_src_ind += 1

        if verbosity == 2:
            print("| | | Iteration: {}".format(iter))
            print("| | | Pixel threshold: {}".format(pixelthresh))
            print("| | | Sources found: {}".format(curr_src_ind))

        if curr_src_ind < nrefinepts:  ###Here think of curr_src_ind representing the amount of sources found this iteration
            div_factor *= 2
            pixelthresh = (img_max-img_median)/div_factor + img_median
        else:
            break

    if debug:
        np.savetxt(debug_report/"pse_metadata_allsources.csv", pse_metadata, delimiter=",")

    if curr_src_ind < nrefinepts:
        ###If max_iters iterations have been performed and we still don't have nrefinepts sources,
        ###we just have to take what we have and do all the heavy calculations for those now.

        if verbosity == 2:
            print("| | done")

        for r in range(num_islands,curr_src_ind):
            prevx = int(pse_metadata_view[r,0])
            prevy = int(pse_metadata_view[r,1])
            kernelsum = 0
            for s in range(kernelsize):
                kernelsum += img_view[(prevy+kernelcircle_view[s,1])-1,(prevx+kernelcircle_view[s,0])-1]
            kernelsum -= bgmap_view[prevy-1,prevx-1]*kernelsize
            pse_metadata_view[r,2] = kernelsum

        pse_metadata_sorted = pse_metadata[pse_metadata[:,2].argsort()][::-1]   ###sort pse_metadata by brightness
        pse_metadata[:] = pse_metadata_sorted
        
        if debug:
            np.savetxt(debug_report/"pse_metadata_allsources_sorted.csv", pse_metadata, delimiter=",")
        num_zeros = nrefinepts - curr_src_ind      #If not all nrefinepts slots are filled, there will be leftover zeros that need to be pushed to the back of the array
        for rr in range(num_islands, nrefinepts):
            if pse_metadata_view[rr,2] < 0:
                pse_metadata_view[rr-num_zeros,:] = pse_metadata_view[rr,:]
                pse_metadata_view[rr,:] = 0
        if debug:
            np.savetxt(debug_report/"pse_metadata_allsources_sorted_zerosmoved.csv", pse_metadata, delimiter=",")

        for r2 in range(num_islands, curr_src_ind):
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

    elif curr_src_ind >= nrefinepts:

        if verbosity == 2:
            print("| | done")

        pse_metadata_sorted = pse_metadata[pse_metadata[:,2].argsort()][::-1]   ###sort pse_metadata by brightness
        pse_metadata[:] = pse_metadata_sorted

        if debug:
            np.savetxt(debug_report/"pse_metadata_allsources_sorted.csv", pse_metadata, delimiter=",")
        
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
            findCentroids(prevx, prevy, kernelrad, img_view, bgmap_view[prevy-1,prevx-1], cntdresults_view)
            pse_metadata_view[t,0] = cntdresults_view[0]   #x centroid
            pse_metadata_view[t,1] = cntdresults_view[1]   #y centroid

    return curr_src_ind

@printEvent
def PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, pixelradius, shape, srcindexmap_initial, srcindexmap_refine, pse_metadata, debug_report, filepath, verbosity, debug):

    num_psesources = 0

    leftmostsafe_x = 1 + sourcesep
    rightmostsafe_x = img_xmax - sourcesep
    topmostsafe_y = 1 + sourcesep
    bottommostsafe_y = img_ymax - sourcesep
    roi = np.ones((img_ymax,img_xmax),dtype=int)

    if verbosity == 2:
        print("| Creating ROI map...")
    createROI(roi, img_xmax, img_ymax, pixelradius, shape)
    if verbosity == 2:
        print("| done")
    if debug:
        if shape == 'rectangle':
            colors = (['gold'])
        elif shape == 'circle':
            colors = (['navy','gold'])
        debuggerPlot(roi, debug_report, "ROI.png", title="ROI", dscrp="Region to be scanned by PSE. Specified by the options shape and buffer.", colors=colors, boundaries=[0,1,2])

    bgmap = np.zeros((img_ymax,img_xmax),dtype=np.double)

    if verbosity == 2:
        print("| Creating background map...")
    fillBackgroundMap(bgmap, img, img_xmax, img_ymax, leftmostsafe_x, rightmostsafe_x, topmostsafe_y, bottommostsafe_y, sourcesep)
    if verbosity == 2:
        print("| done")
    if debug:
        debuggerPlot(bgmap, debug_report, "bgmap.png", figsize=(8,7), vmin=np.percentile(bgmap,1), vmax=np.percentile(bgmap,99), colbar=True, colbar_extend=True, title="Background map", dscrp="Used for determining the amplitude-above-background for each pixel. \nThe PSE does not scan pixels in a thin ({}-pixel-wide) strip around the \nborder, so the background is not calculated there.".format(sourcesep), textpos=0.07)

    curr_src_ind = 0
    occupymap = -1*np.ones(img.shape, dtype=int)

    if verbosity >= 1:
        print("| Finding saturation islands...")
    curr_src_ind = mapSaturationIslands(img, img_xmax, img_ymax, leftmostsafe_x, rightmostsafe_x, topmostsafe_y, bottommostsafe_y, roi, pse_metadata, occupymap, srcindexmap_initial, srcindexmap_refine, sourcesep, pixsat, npts, nrefinepts, curr_src_ind, debug_report, verbosity, debug)     #return curr_src_ind to use in mapRemainingSources
    if curr_src_ind < nrefinepts:
        if verbosity >= 1:
            print("| | Found {} saturation islands.".format(curr_src_ind))
    elif curr_src_ind >= nrefinepts:
        if verbosity >= 1:
            print("| | Found {} saturation islands. Trimmed to the {} brightest.".format(curr_src_ind,nrefinepts))
        if verbosity == 2:
            print("| | Centroids and brightnesses placed in pse_metadata.")
            print("| | {} brightest kernels drawn to srcindexmap_initial.".format(npts))
        if debug:
            debuggerPlot(srcindexmap_initial, debug_report, "srcindexmap_initial.png", title="srcindexmap_initial", dscrp="The occupied source kernels for the top {} brightest sources \nin the image. Used to find an initial WCS solution.".format(npts), textpos=0.07, colors=["blue","yellow"], boundaries=[-1,0,npts])
        if verbosity == 2:
            print("| | {} brightest kernels drawn to srcindexmap_refine.".format(nrefinepts))
        if debug:
            debuggerPlot(srcindexmap_refine, debug_report, "srcindexmap_refine.png", title="srcindexmap_refine", dscrp="The occupied source kernels for the top {} brightest sources \nin the image. Used to refine the WCS solution.".format(nrefinepts), textpos=0.07, colors=["blue","yellow"], boundaries=[-1,0,nrefinepts])
        if verbosity >= 1:
            print("| done")
        num_psesources = nrefinepts
        return num_psesources

    kernelsize = getCircleSize(kernelrad)
    kernelcircle = np.zeros((kernelsize,2),dtype=int)
    makePixelCircle(kernelrad,kernelcircle)

    sourcesepsize = getCircleSize(sourcesep)
    sourcesepcircle = np.zeros((sourcesepsize,2),dtype=int)
    makePixelCircle(sourcesep,sourcesepcircle)

    img_median = np.median(img)
    img_max = np.max(img)
    
    if verbosity ==2:
        print("| Median amplitude: {}".format(img_median))
        print("| Max amplitude: {}".format(img_max))

    if verbosity >= 1:
        print("| Finding remaining sources...")
    curr_src_ind = mapRemainingSources(img, img_xmax, img_ymax, img_max, img_median, leftmostsafe_x, rightmostsafe_x, topmostsafe_y, bottommostsafe_y, roi, pse_metadata, bgmap, srcindexmap_initial, srcindexmap_refine, occupymap, kernelrad, kernelcircle, kernelsize, sourcesep, sourcesepcircle, sourcesepsize, npts, nrefinepts, curr_src_ind, debug_report, verbosity, debug)
    if curr_src_ind < nrefinepts:
        if verbosity >= 1:
            print("| | Unsuccessful at finding {} sources. Found {} sources.".format(nrefinepts, curr_src_ind))
        if verbosity == 2:
            print("| | Centroids and brightnesses placed in pse_metadata.")
            print("| | {} brightest kernels drawn to srcindexmap_initial.".format(npts))
        if debug:
            debuggerPlot(srcindexmap_initial, debug_report, "srcindexmap_initial.png", title="srcindexmap_initial", dscrp="The occupied source kernels for the top {} brightest sources \nin the image. Used to find an initial WCS solution.".format(npts), textpos=0.07, colors=["blue","yellow"], boundaries=[-1,0,npts])
        if verbosity == 2:
            print("| | {} brightest kernels drawn to srcindexmap_refine.".format(curr_src_ind))
        if debug:
            debuggerPlot(srcindexmap_refine, debug_report, "srcindexmap_refine.png", title="srcindexmap_refine", dscrp="The occupied source kernels for the top {} brightest sources \nin the image. Used to refine the WCS solution.".format(curr_src_ind), textpos=0.07, colors=["blue","yellow"], boundaries=[-1,0,curr_src_ind])
        if verbosity >= 1:
            print("| done")
        num_psesources = curr_src_ind
    elif curr_src_ind >= nrefinepts:
        if verbosity >= 1:
            print("| | Found {} sources total. Trimmed to {} sources.".format(curr_src_ind, nrefinepts))
        if verbosity == 2:
            print("| | Centroids and brightnesses placed in pse_metadata.")
            print("| | {} brightest kernels drawn to srcindexmap_initial.".format(npts))
        if debug:
            debuggerPlot(srcindexmap_initial, debug_report, "srcindexmap_initial.png", title="srcindexmap_initial", dscrp="The occupied source kernels for the top {} brightest sources \nin the image. Used to find an initial WCS solution.".format(npts), textpos=0.07, colors=["blue","yellow"], boundaries=[-1,0,npts])
        if verbosity == 2:
            print("| | {} brightest kernels drawn to srcindexmap_refine.".format(nrefinepts))
        if debug:
            debuggerPlot(srcindexmap_refine, debug_report, "srcindexmap_refine.png", title="srcindexmap_refine", dscrp="The occupied source kernels for the top {} brightest sources \nin the image. Used to refine the initial WCS solution.".format(nrefinepts), textpos=0.07, colors=["blue","yellow"], boundaries=[-1,0,nrefinepts])
        if verbosity >= 1:
            print("| done")
        num_psesources = nrefinepts

    if debug:
        np.savetxt(debug_report/"pse_metadata.csv", pse_metadata, delimiter=",")

    if debug:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits
        plt.figure(figsize=(10,8))
        image_data = fits.getdata(filepath)
        plt.subplots_adjust(bottom=0.15)
        plt.imshow(image_data, cmap="gray", norm=LogNorm())
        plt.scatter(pse_metadata[:,0], pse_metadata[:,1], color='red', marker='.')
        plt.title("pse_metadata centroids")
        dscrp="Centroids from the pse_metadata, plotted overtop the image."
        plt.figtext(0.5, 0.05, dscrp, ha="center", fontsize=9)
        plt.savefig(debug_report/"pse_metadata_centroids.png")
        plt.show()

    return num_psesources

                        
    


