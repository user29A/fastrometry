from astroquery.gaia import Gaia
import scipy.optimize as optimization
import numpy as np
from pathlib import Path
import csv

cimport numpy as np
from libc.math cimport sin, cos, pi, asin, acos, atan, atan2, sqrt, abs

cdef getSkyCoordsFromPixCoord(double x, double y, double CD1_1, double CD1_2, double CD2_1, double CD2_2, double CRPIX1, double CRPIX2, double a0, double d0, double[:] skycoord_view):
    
    """
    This function is not used in the program, but is put here for illustrating how any point (x,y) in the original image may be transformed
    into sky coordinates after the WCS algorithm has generated a solution (the CD matrix and the CRPIX reference point).
    """

    X = CD1_1*(x-CRPIX1) + CD1_2*(y-CRPIX2)
    Y = CD2_1*(x-CRPIX1) + CD2_2*(y-CRPIX2)

    # The inverse gnomonic equations from the paper (eqs. 3) have an error in the equation for the declination. They are given below.
    # skycoord_view[0] = a0 + atan2(X,cos(d0)-Y*sin(d0))
    # skycoord_view[1] = asin(sin(d0)+Y*cos(d0))/sqrt(1+X*X+Y*Y)
    # Using an alternate set of formulas from https://mathworld.wolfram.com/GnomonicProjection.html:

    p = sqrt(X*X+Y*Y)
    c = atan(p)
    skycoord_view[0] = a0 + atan2(X*sin(c),p*cos(d0)*cos(c)-Y*sin(d0)*sin(c))
    skycoord_view[1] = asin(cos(c)*sin(d0)+Y*sin(c)*cos(d0)/p)

def inverseTransformPointUsingCD(CD11, CD12, CD21, CD22, CRPIXx, CRPIXy, intrmx, intrmy, invtransf_intrmpoint_view):
    
    """
    Use the 4 CD matrix elements and the 2 CRPIX reference coordinates to inverse-transform individual intermediate points.
    """

    determinant = CD11*CD22-CD12*CD21
    CD11inv = 1/determinant*CD22
    CD12inv = -1/determinant*CD12
    CD21inv = -1/determinant*CD21
    CD22inv = 1/determinant*CD11
    invtransf_intrmpoint_view[0] = CRPIXx + CD11inv*intrmx + CD12inv*intrmy
    invtransf_intrmpoint_view[1] = CRPIXy + CD21inv*intrmx + CD22inv*intrmy

def fitNrefineptsToGetCD(params, matches_arr, matches):
    """
    The optimizer function, for which the CD matrix and the CRPIX point are the parameters. After many runs of this function, the optimizer
    will settle on the CD elements and CRPIX coordinates which minimize the Nrefinepts residuals. Each residual is the difference between the forward-
    transformed pse coordinate and the intermediate coordinate that should result.
    """

    CD11 = params[0]
    CD12 = params[1]
    CD21 = params[2]
    CD22 = params[3]
    CRPIXx = params[4]
    CRPIXy = params[5]

    residual_arr = np.zeros(matches*2)

    for s in range(matches):
        intrmx = matches_arr[s,0]
        intrmy = matches_arr[s,1]
        psex = matches_arr[s,2]
        psey = matches_arr[s,3]
        residual_arr[2*s] = (CD11*(psex-CRPIXx) + CD12*(psey-CRPIXy)) - intrmx
        residual_arr[2*s+1] = (CD21*(psex-CRPIXx) + CD22*(psey-CRPIXy)) - intrmy

    return residual_arr

cdef void refineSolution(double[:,:] allintrmcoords_view, int num_catsources, double[:,:] smetadata_view, int num_psesources, int npts, int nrefinepts, int img_xmax, int img_ymax, int[:,:] sindmap_refine_view, double S, double phi, double CRPIXx, double CRPIXy, double[:] solution_refined, str user_dir, str filename, int verbose, int debug):
    
    """
    Test previous solution using as many points as possible (all num_psesources points from the PSE and all num_catsources points from the catalogue).
    Instead of optimizing the parameters S, phi, CRPIXx and CRPIXy, simply optimize the individual matrix elements (labelled CD11, CD12, CD21, and CD22).
    Since the matrix elements no longer depend on each other, this allows for some skew in the inverse-transformation. Therefore if the image has skew, it
    will be captured in the optimizer solution (which consists of the parameters CD11, CD12, CD21, CD22, CRPIXx, CRPIXy). 
    """

    cdef:
        int nrefinepts_intrm
        int nrefinepts_pse
    
    # As mentioned before, it's possible (but rare) that the PSE output or the catalogue output may contain
    # less than npts in total. If this is the case, all possible points (for one or both of them) have already 
    # been tested in the initial solution. So using "all" test points would do nothing new (if both are less
    # than ntps) or increase only one set, which is not beneficial since we are looking for new intermediate 
    # points that land on new pse points. In this unfortunate circumstance we terminate the function with a 
    # void return.

    if num_catsources < npts or num_psesources < npts:
        return
    else:
        nrefinepts_intrm = num_catsources
        nrefinepts_pse = num_psesources

    cdef:
        double[:,:] psecoords_view = smetadata_view[:nrefinepts_pse,:2]
        double[:,:] intrmcoords_view = allintrmcoords_view[:nrefinepts_intrm,:]
        np.ndarray[dtype=np.double_t,ndim=2] matches_arr = np.zeros((nrefinepts_intrm,4))
        double[:,:] matches_view = matches_arr
        double CD11
        double CD11_guess = S*cos(phi)
        double CD12
        double CD12_guess = -S*sin(phi)
        double CD21
        double CD21_guess = S*sin(phi)
        double CD22
        double CD22_guess = S*cos(phi)
        double CRPIXx_guess = CRPIXx
        double CRPIXy_guess = CRPIXy
        double[:] invtransf_intrmpoint_view = np.empty(2,dtype=np.double)
        int src_ind
        Py_ssize_t k
        Py_ssize_t m

    guess = np.array([CD11_guess, CD12_guess, CD21_guess, CD22_guess, CRPIXx_guess, CRPIXy_guess])

    if debug >= 1:
        invtransf_xs = []
        invtransf_ys = []
        matchintrmxs = []
        matchintrmys = []

    matches = 0
    for k in range(nrefinepts_intrm):
        inverseTransformPointUsingSphi(S, phi, CRPIXx, CRPIXy, intrmcoords_view[k,0], intrmcoords_view[k,1], invtransf_intrmpoint_view)
        if debug >= 1:
            invtransf_xs.append(invtransf_intrmpoint_view[0])
            invtransf_ys.append(invtransf_intrmpoint_view[1])
        if 1 <= invtransf_intrmpoint_view[0] <= img_xmax and 1 <= invtransf_intrmpoint_view[1] <= img_ymax:
            src_ind = sindmap_refine_view[int(invtransf_intrmpoint_view[1])-1,int(invtransf_intrmpoint_view[0])-1]
            if src_ind != -1:
                matches_view[matches,0] = intrmcoords_view[k,0]
                matches_view[matches,1] = intrmcoords_view[k,1]
                matches_view[matches,2] = psecoords_view[src_ind,0]
                matches_view[matches,3] = psecoords_view[src_ind,1]
                matches += 1
                if debug >= 1:
                    matchintrmxs.append(invtransf_intrmpoint_view[0])
                    matchintrmys.append(invtransf_intrmpoint_view[1])

    if verbose >= 1:
        print(matches)
    
    if debug >= 1:

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits

        image_data = fits.getdata('{}\\{}'.format(user_dir,filename))
        fig5 = plt.figure(5,figsize=(10,8))
        axes5 = fig5.add_subplot(111)
        axes5.imshow(image_data, cmap="gray", norm=LogNorm())
        axes5.scatter(matchintrmxs,matchintrmys,color='pink')
        axes5.scatter(psecoords_view[:,0],psecoords_view[:,1],marker="+",color='y')
        axes5.scatter(invtransf_xs,invtransf_ys,c='red',marker='.')
        plt.show()
        plt.close()
    
    matches_trimmed = matches_arr[:matches,:]
    optimizedparams = optimization.least_squares(fun=fitNrefineptsToGetCD, x0=guess, args=(matches_trimmed, matches))
    CD11 = optimizedparams.x[0]
    CD12 = optimizedparams.x[1]
    CD21 = optimizedparams.x[2]
    CD22 = optimizedparams.x[3]
    CRPIXx = optimizedparams.x[4]
    CRPIXy = optimizedparams.x[5]
    solution_refined[0] = CD11
    solution_refined[1] = CD12
    solution_refined[2] = CD21
    solution_refined[3] = CD22
    solution_refined[4] = CRPIXx
    solution_refined[5] = CRPIXy

    if debug >= 1:

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits

        newinvtransf_xs = []
        newinvtransf_ys = []
        newmatchintrmxs = []
        newmatchintrmys = []
        newmatches = 0
        for k in range(nrefinepts_intrm):
            inverseTransformPointUsingCD(CD11, CD12, CD21, CD22, CRPIXx, CRPIXy, intrmcoords_view[k,0], intrmcoords_view[k,1], invtransf_intrmpoint_view)
            newinvtransf_xs.append(invtransf_intrmpoint_view[0])
            newinvtransf_ys.append(invtransf_intrmpoint_view[1])
            if 1 <= invtransf_intrmpoint_view[0] <= img_xmax and 1 <= invtransf_intrmpoint_view[1] <= img_ymax:
                src_ind = sindmap_refine_view[int(invtransf_intrmpoint_view[1])-1,int(invtransf_intrmpoint_view[0])-1]
                if src_ind != -1:
                    newmatches += 1
                    newmatchintrmxs.append(invtransf_intrmpoint_view[0])
                    newmatchintrmys.append(invtransf_intrmpoint_view[1])
        print(newmatches)
        fig6 = plt.figure(6,figsize=(10,8))
        axes6 = fig6.add_subplot(111)
        axes6.imshow(image_data, cmap="gray", norm=LogNorm())
        axes6.scatter(newmatchintrmxs,newmatchintrmys,color='pink')
        axes6.scatter(psecoords_view[:,0],psecoords_view[:,1],marker="+",color='y')
        axes6.scatter(newinvtransf_xs,newinvtransf_ys,c='red',marker='.')
        plt.show()
        plt.close()

cdef inverseTransformPointUsingSphi(double S, double phi, double CRPIXx, double CRPIXy, double intrmx, double intrmy, double[:] invtransf_intrmpoint_view):
    """
    Use S, phi and the 2 CRPIX reference coordinates to inverse-transform individual intermediate points.
    """

    invtransf_intrmpoint_view[0] = CRPIXx + (intrmx*cos(phi) + intrmy*sin(phi))/S
    invtransf_intrmpoint_view[1] = CRPIXy + (-intrmx*sin(phi) + intrmy*cos(phi))/S     

cdef inverseTransformTriangleUsingSphi(double S, double phi, double CRPIXx, double CRPIXy, double intrmAx, double intrmAy, double intrmBx, double intrmBy, double intrmCx, double intrmCy, double[:,:] invtransf_intrmtriangle_view):
    """
    Use S, phi and the 2 CRPIX reference coordinates to inverse-transform intermediate triangles (triplets of intermediate points).
    """

    invtransf_intrmtriangle_view[0,0] = CRPIXx + (intrmAx*cos(phi) + intrmAy*sin(phi))/S
    invtransf_intrmtriangle_view[0,1] = CRPIXy + (-intrmAx*sin(phi) + intrmAy*cos(phi))/S
    invtransf_intrmtriangle_view[1,0] = CRPIXx + (intrmBx*cos(phi) + intrmBy*sin(phi))/S
    invtransf_intrmtriangle_view[1,1] = CRPIXy + (-intrmBx*sin(phi) + intrmBy*cos(phi))/S
    invtransf_intrmtriangle_view[2,0] = CRPIXx + (intrmCx*cos(phi) + intrmCy*sin(phi))/S
    invtransf_intrmtriangle_view[2,1] = CRPIXy + (-intrmCx*sin(phi) + intrmCy*cos(phi))/S

def fitTrianglesToGetSphi(params, pseAx, pseAy, pseBx, pseBy, pseCx, pseCy, intrmAx, intrmAy, intrmBx, intrmBy, intrmCx, intrmCy):
    """
    The optimizer function, for which S, phi, CRPIXx and CRPIXy are the parameters. After many runs of this function, the optimizer
    will settle on the S, phi, CRPIXx and CRPIXy which minimize the 6 residuals. Each residual is the difference between the forward-
    transformed pse coordinate and the intermediate coordinate that should result. (Here however it is written the other way around.)
    """

    S = params[0]
    phi = params[1]
    CRPIXx = params[2]
    CRPIXy = params[3]
    residualAx = (S*cos(phi)*(pseAx-CRPIXx) - S*sin(phi)*(pseAy-CRPIXy)) - intrmAx
    residualAy = (S*sin(phi)*(pseAx-CRPIXx) + S*cos(phi)*(pseAy-CRPIXy)) - intrmAy
    residualBx = (S*cos(phi)*(pseBx-CRPIXx) - S*sin(phi)*(pseBy-CRPIXy)) - intrmBx
    residualBy = (S*sin(phi)*(pseBx-CRPIXx) + S*cos(phi)*(pseBy-CRPIXy)) - intrmBy
    residualCx = (S*cos(phi)*(pseCx-CRPIXx) - S*sin(phi)*(pseCy-CRPIXy)) - intrmCx
    residualCy = (S*sin(phi)*(pseCx-CRPIXx) + S*cos(phi)*(pseCy-CRPIXy)) - intrmCy
    return np.array([residualAx, residualAy, residualBx, residualBy, residualCx, residualCy])

cdef getTriangleData(double x1, double y1, double x2, double y2, double x3, double y3, triangle_datalist):

    """
    Given 6 coordinates (3 points), this function determines labels for the three points (A for
    the point with the largest vertex measure, B for the middle, and C for the smallest) and
    calculates the longest sidelength (BC) as well as the angle of the vector that points from B 
    to C (relative to the x-axis of the image). All of these geometric properties are stored in a
    10-long array that itself will be inserted into a triangle "database" to be accessed later.
    """

    cdef:
        double sidelength12
        double sidelength23
        double sidelength31
        double pointAx
        double pointAy
        double pointBx
        double pointBy
        double pointCx
        double pointCy
        double sidelengthAB
        double sidelengthBC
        double sidelengthCA
        double vertexA
        double vertexB
        double vertexC
        double fieldvectorangle

    sidelength12 = sqrt((x1-x2)**2+(y1-y2)**2)
    sidelength23 = sqrt((x2-x3)**2+(y2-y3)**2)
    sidelength31 = sqrt((x3-x1)**2+(y3-y1)**2)

    if sidelength12 < sidelength31 < sidelength23:
        pointAx = x1
        pointAy = y1
        pointBx = x2
        pointBy = y2
        pointCx = x3
        pointCy = y3
    elif sidelength31 < sidelength12 < sidelength23:
        pointAx = x1
        pointAy = y1
        pointBx = x3
        pointBy = y3
        pointCx = x2
        pointCy = y2
    elif sidelength12 < sidelength23 < sidelength31:
        pointAx = x2
        pointAy = y2
        pointBx = x1
        pointBy = y1
        pointCx = x3
        pointCy = y3
    elif sidelength23 < sidelength12 < sidelength31:
        pointAx = x2
        pointAy = y2
        pointBx = x3
        pointBy = y3
        pointCx = x1
        pointCy = y1
    elif sidelength31 < sidelength23 < sidelength12:
        pointAx = x3
        pointAy = y3
        pointBx = x1
        pointBy = y1
        pointCx = x2
        pointCy = y2
    elif sidelength23 < sidelength31 < sidelength12:
        pointAx = x3
        pointAy = y3
        pointBx = x2
        pointBy = y2
        pointCx = x1
        pointCy = y1

    sidelengthAB = sqrt((pointAx-pointBx)**2+(pointAy-pointBy)**2)
    sidelengthBC = sqrt((pointBx-pointCx)**2+(pointBy-pointCy)**2)
    sidelengthCA = sqrt((pointCx-pointAx)**2+(pointCy-pointAy)**2)
    vertexA = acos((sidelengthAB**2+sidelengthCA**2-sidelengthBC**2)/(2*sidelengthAB*sidelengthCA))
    vertexB = acos((sidelengthAB**2+sidelengthBC**2-sidelengthCA**2)/(2*sidelengthAB*sidelengthBC))
    vertexC = acos((sidelengthCA**2+sidelengthBC**2-sidelengthAB**2)/(2*sidelengthCA*sidelengthBC))
    fieldvectorangle = atan2((pointCy-pointBy),(pointCx-pointBx))

    triangle_datalist[0] = pointAx
    triangle_datalist[1] = pointAy
    triangle_datalist[2] = pointBx
    triangle_datalist[3] = pointBy
    triangle_datalist[4] = pointCx
    triangle_datalist[5] = pointCy
    triangle_datalist[6] = vertexB
    triangle_datalist[7] = vertexC
    triangle_datalist[8] = sidelengthBC
    triangle_datalist[9] = fieldvectorangle

cdef generateTriangles(int npts, double[:,:] coords_view, double[:,:] triangles_view):
    """
    The a,b,c triple-nested for-loop iterates over every possible 3-combination of points (intermediate or
    PSE points, depending on the function input). For each combination (triangle), getTriangleData calculates 
    some associated geometrical properties (metadata), and then an array of this metadata is placed in a 
    "repository" of triangle metadata arrays. This function is called twice in total, and will populate two 
    (intermediate and PSE) triangle repositories. The metadata stored inside both will be accessed at a later 
    point, when triangles are comparing against each other in the i,j double-nested for-loop.
    """

    cdef:
        Py_ssize_t triangleNo = 0
        Py_ssize_t a
        Py_ssize_t b
        Py_ssize_t c
        Py_ssize_t propertyNo
        double[:] triangledata_view = np.empty(10,dtype=np.double)

    triangleNo = 0
    for a in range(0,npts-2):
        for b in range(a+1,npts-1):
            for c in range(b+1,npts):
                getTriangleData(coords_view[a,0],coords_view[a,1],coords_view[b,0],coords_view[b,1],coords_view[c,0],coords_view[c,1], triangledata_view)
                for propertyNo in range(10):
                    triangles_view[triangleNo,propertyNo] = triangledata_view[propertyNo]
                triangleNo += 1

cdef void findInitialSolution(double[:,:] allintrmcoords_view, int num_catsources, double[:,:] smetadata_view, int num_psesources, int[:,:] sindmap_initial_view, int npts, int img_xmax, int img_ymax, double scale, double scalebnds, object rotation, object rotationbnds, double vertextol, int minmatches, int kerneldiam, double[:] solution_initial, str user_dir, str filename, int verbose, int debug):
    """
    """

    cdef:
        int npts_intrm
        int npts_pse
    
    # It's possible (but rare) that the PSE output or the catalogue output may contain less than npts in total.
    # In that case, we have to use however many points are available.

    if num_catsources < npts:
        npts_intrm = num_catsources
    else:
        npts_intrm = npts
    
    if num_psesources < npts:
        npts_pse = num_psesources
    else:
        npts_pse = npts

    cdef:
        double[:,:] psecoords_view = smetadata_view[:npts_pse,:2]
        double[:,:] intrmcoords_view = allintrmcoords_view[:npts_intrm,:]
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t p
        int nchoose3_pse = int(npts_pse*(npts_pse-1)*(npts_pse-2)/6)
        int nchoose3_intrm = int(npts_intrm*(npts_intrm-1)*(npts_intrm-2)/6)
        double S
        double S_guess
        double S_lb
        double S_ub
        bint rotation_provided
        bint rotationbnds_provided
        double phi
        double phi_guess
        double phi_lb
        double phi_ub
        double CRPIXx
        double CRPIXx_guess
        double CRPIXx_lb
        double CRPIXx_ub
        double CRPIXy
        double CRPIXy_guess
        double CRPIXy_lb
        double CRPIXy_ub
        double sumx
        double sumy
        double sidelengthtol
        int matches
        double[:,:] psetriangles_view = np.empty((nchoose3_pse,10),dtype=np.double)
        double[:,:] intrmtriangles_view = np.empty((nchoose3_intrm,10),dtype=np.double)
        double intrmlengthBC_lb
        double intrmlengthBC_ub
        double[:,:] invtransf_intrmtriangle_view = np.empty((3,2),dtype=np.double)
        double[:] invtransf_intrmpoint_view = np.empty(2,dtype=np.double)

    if debug >= 1:
        print("psecoords_view",np.asarray(psecoords_view))
        print("intrmcoords_view",np.asarray(intrmcoords_view))

    if debug >= 1:
        psecoords_xrange = np.ptp(psecoords_view[:,0])
        intrmcoords_xrange = np.ptp(intrmcoords_view[:,0])
        approx_scale = intrmcoords_xrange/psecoords_xrange
        print("approx_scale ",approx_scale)

    S_guess = scale
    S_lb = scale/(1+scalebnds)
    S_ub = scale*(1+scalebnds)

    if debug >= 1:
        print("S_guess ",S_guess)

    if rotation is None and rotationbnds is None:
        rotation_provided = False
        rotationbnds_provided = False
        phi_guess = 0       ###these variables will be properly declared later
        phi_lb = 0
        phi_ub = 0
    elif rotation is not None and rotationbnds is not None:
        rotation_provided = True
        rotationbnds_provided = True
        phi_guess = rotation
        phi_lb = rotation-rotationbnds-vertextol
        phi_ub = rotation+rotationbnds+vertextol
    elif rotation is None and rotationbnds is not None:
        rotation_provided = False
        rotationbnds_provided = True
        phi_guess = 0
        phi_lb = 0-rotationbnds-vertextol
        phi_ub = 0+rotationbnds+vertextol
    elif rotation is not None and rotationbnds is None:
        rotation_provided = True
        rotationbnds_provided = False
        phi_guess = rotation
        phi_lb = rotation-10*pi/180-vertextol
        phi_ub = rotation+10*pi/180+vertextol    

    sumx = 0
    sumy = 0
    for p in range(npts_pse):
        sumx += psecoords_view[p,0]
        sumy += psecoords_view[p,1]
    CRPIXx_guess = sumx/npts_pse
    CRPIXy_guess = sumy/npts_pse
    CRPIXx_lb = 1
    CRPIXx_ub = img_xmax
    CRPIXy_lb = 1
    CRPIXy_ub = img_ymax

    if debug >= 1:
        print("phi_guess ",phi_guess)
    
    guess = np.array([S_guess, phi_guess, CRPIXx_guess, CRPIXy_guess])
    bnds = np.array([[S_lb,phi_lb,CRPIXx_lb,CRPIXy_lb],[S_ub,phi_ub,CRPIXx_ub,CRPIXy_ub]])
    
    if debug >= 1:
        print("guess ",guess)
        print("bnds[0] bnds[1]",bnds[0],bnds[1])

    if debug >= 1:
        print("first ten pse xs ",np.asarray(psecoords_view)[:10,0])
        print("first ten pse ys ",np.asarray(psecoords_view)[:10,1])

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits

        image_data = fits.getdata('{}\\{}'.format(user_dir,filename))

        fig1 = plt.figure(1,figsize=(10,8))
        axes1 = fig1.add_subplot(111)
        axes1.imshow(image_data, cmap="gray", norm=LogNorm())
        axes1.scatter(np.asarray(psecoords_view)[:10,0], np.asarray(psecoords_view)[:10,1], marker=".", c="blue")
        axes1.scatter(np.asarray(intrmcoords_view)[:10,0]/approx_scale+CRPIXx_guess,np.asarray(intrmcoords_view)[:10,1]/approx_scale+CRPIXy_guess, marker=".", c="red")
        plt.show()

        fig2 = plt.figure(2,figsize=(10,8))
        axes2 = fig2.add_subplot(111)
        axes2.imshow(image_data, cmap="gray", norm=LogNorm())
        axes2.scatter(np.asarray(psecoords_view)[:,0], np.asarray(psecoords_view)[:,1], marker=".", c="blue")
        axes2.scatter(np.asarray(intrmcoords_view)[:,0]/approx_scale+CRPIXx_guess,np.asarray(intrmcoords_view)[:,1]/approx_scale+CRPIXy_guess, marker=".", c="red")
        plt.show()

        fig3 = plt.figure(3,figsize=(10,8))
        fig3 = fig3.add_subplot(111)
        fig3.imshow(image_data, cmap="gray", norm=LogNorm())
        fig3.scatter(np.asarray(smetadata_view[:,:2])[:,0], np.asarray(smetadata_view[:,:2])[:,1], marker=".", c="blue")
        fig3.scatter(np.asarray(allintrmcoords_view[:,:])[:,0]/approx_scale+CRPIXx_guess, np.asarray(allintrmcoords_view[:,:])[:,1]/approx_scale+CRPIXy_guess, marker=".", c="red")
        plt.show()

    generateTriangles(npts_pse, psecoords_view, psetriangles_view)
    generateTriangles(npts_intrm, intrmcoords_view, intrmtriangles_view)

    for i in range(nchoose3_pse):
        if debug >= 1:
            print("i ",i)
        psesrcindex_A = sindmap_initial_view[int(psetriangles_view[i,1])-1,int(psetriangles_view[i,0])-1]
        psesrcindex_B = sindmap_initial_view[int(psetriangles_view[i,3])-1,int(psetriangles_view[i,2])-1]
        psesrcindex_C = sindmap_initial_view[int(psetriangles_view[i,5])-1,int(psetriangles_view[i,4])-1]

        for j in range(nchoose3_intrm):
            if abs(psetriangles_view[i,6]-intrmtriangles_view[j,6]) < vertextol:
                if abs(psetriangles_view[i,7]-intrmtriangles_view[j,7]) < vertextol:
                    if intrmtriangles_view[j,8] > S_lb*(psetriangles_view[i,8]-kerneldiam) and intrmtriangles_view[j,8] < S_ub*(psetriangles_view[i,8]+kerneldiam):
                        theta = intrmtriangles_view[j,9]-psetriangles_view[i,9]
                        if debug >= 1:
                            print("theta ", theta)
                        if rotation_provided == False and rotationbnds_provided == False:            ###if rotation and rotationbnds are not supplied
                            guess[1] = theta
                            bnds[0,1] = theta - vertextol
                            bnds[1,1] = theta + vertextol
                        else:
                            if theta < phi_lb or theta > phi_ub:
                                continue
                        optimizedparams = optimization.least_squares(fun=fitTrianglesToGetSphi, x0=guess, bounds=bnds, args=(psetriangles_view[i,0], psetriangles_view[i,1], psetriangles_view[i,2], psetriangles_view[i,3], psetriangles_view[i,4], psetriangles_view[i,5], intrmtriangles_view[j,0], intrmtriangles_view[j,1], intrmtriangles_view[j,2], intrmtriangles_view[j,3], intrmtriangles_view[j,4], intrmtriangles_view[j,5]))
                        S = optimizedparams.x[0]
                        phi = optimizedparams.x[1]
                        CRPIXx = optimizedparams.x[2]
                        CRPIXy = optimizedparams.x[3]
                        inverseTransformTriangleUsingSphi(S, phi, CRPIXx, CRPIXy, intrmtriangles_view[j,0], intrmtriangles_view[j,1], intrmtriangles_view[j,2], intrmtriangles_view[j,3], intrmtriangles_view[j,4], intrmtriangles_view[j,5], invtransf_intrmtriangle_view)
                        if 1 <= invtransf_intrmtriangle_view[0,0] <= img_xmax and 1 <= invtransf_intrmtriangle_view[0,1] <= img_ymax:
                            if 1 <= invtransf_intrmtriangle_view[1,0] <= img_xmax and 1 <= invtransf_intrmtriangle_view[1,1] <= img_ymax:
                                if 1 <= invtransf_intrmtriangle_view[2,0] <= img_xmax and 1 <= invtransf_intrmtriangle_view[2,1] <= img_ymax:
                                    if psesrcindex_A == sindmap_initial_view[int(invtransf_intrmtriangle_view[0,1])-1,int(invtransf_intrmtriangle_view[0,0])-1]:
                                        if psesrcindex_B == sindmap_initial_view[int(invtransf_intrmtriangle_view[1,1])-1,int(invtransf_intrmtriangle_view[1,0])-1]:
                                            if psesrcindex_C == sindmap_initial_view[int(invtransf_intrmtriangle_view[2,1])-1,int(invtransf_intrmtriangle_view[2,0])-1]:
                                                matches = 0
                                                if debug >= 1:
                                                    matchintrmxs = []
                                                    matchintrmys = []
                                                    invtransf_xs = []
                                                    invtransf_ys = []
                                                for k in range(npts_intrm):
                                                    inverseTransformPointUsingSphi(S, phi, CRPIXx, CRPIXy, intrmcoords_view[k,0], intrmcoords_view[k,1], invtransf_intrmpoint_view)
                                                    if debug >= 1:
                                                        invtransf_xs.append(invtransf_intrmpoint_view[0])
                                                        invtransf_ys.append(invtransf_intrmpoint_view[1])
                                                    if 1 <= invtransf_intrmpoint_view[0] <= img_xmax and 1 <= invtransf_intrmpoint_view[1] <= img_ymax:
                                                        if sindmap_initial_view[int(invtransf_intrmpoint_view[1])-1,int(invtransf_intrmpoint_view[0])-1] != -1:
                                                            matches += 1
                                                            if debug >= 1:
                                                                matchintrmxs.append(invtransf_intrmpoint_view[0])
                                                                matchintrmys.append(invtransf_intrmpoint_view[1])
                                                if matches >= minmatches:
                                                    if debug >= 1:
                                                        from math import degrees
                                                        print("-- GOT A SOLUTION --")
                                                        print("S: ",S)
                                                        print("phi raw: ",phi)
                                                        print("phi in degrees: ",degrees(phi))
                                                        print("CRPIXx: ",CRPIXx)
                                                        print("CRPIXy: ",CRPIXy)
                                                        print("number of matches",matches)
                                                    solution_initial[0] = S
                                                    solution_initial[1] = phi
                                                    solution_initial[2] = CRPIXx
                                                    solution_initial[3] = CRPIXy

                                                    if debug >= 1:
                                                        import matplotlib.pyplot as plt
                                                        from matplotlib.colors import LogNorm
                                                        from astropy.io import fits

                                                        fig4 = plt.figure(4,figsize=(10,8))
                                                        axes4 = fig4.add_subplot(111)
                                                        axes4.imshow(image_data, cmap="gray", norm=LogNorm())
                                                        axes4.scatter(matchintrmxs,matchintrmys,color='pink')
                                                        axes4.scatter(psecoords_view[:,0],psecoords_view[:,1],marker="+",color='y')
                                                        axes4.scatter(invtransf_xs,invtransf_ys,c='red',marker='.')
                                                        plt.show()
                                                        plt.close()

                                                    return
    print("No solution found")
    return

def getIntermediateCoords(ra, dec, fieldradius, fieldquery, catalogue, filter, nrefinepts, intrmcoords, meancatcoords, user_dir, verbose, debug):

    if catalogue == "GaiaDR3":
            
        if filter == "rp":
            filtkey = "phot_rp_mean_mag"
        elif filter == "g":
            filtkey = "phot_g_mean_mag"
        elif filter == "bp":
            filtkey = "phot_bp_mean_mag"

        if not Path('{}\\gaiaqueries'.format(user_dir)).is_dir():
            if verbose >= 1:
                print("Creating {}\\gaiaqueries".format(user_dir))
            Path('{}\\gaiaqueries'.format(user_dir)).mkdir(parents=True)
        filenameifexists = '{}\\gaiaqueries\\gaiatable_ra{:.9}_dec{:.9}_{}_{}_{}_{}.csv'.format(user_dir, ra, dec, fieldquery, catalogue, filter, nrefinepts)
        if Path(filenameifexists).is_file():
            if verbose >= 1:
                print("opening existing table: {}".format(filenameifexists))
            with open('{}\\gaiaqueries\\gaiatable_ra{:.9}_dec{:.9}_{}_{}_{}_{}.csv'.format(user_dir, ra, dec, fieldquery, catalogue, filter, nrefinepts), 'r') as f:
                f.readline()        ###throw away header
                csvfile = csv.reader(f)
                ras = np.zeros(nrefinepts)
                decs = np.zeros(nrefinepts)
                mags = np.zeros(nrefinepts)
                for r,row in enumerate(csvfile):
                    ras[r] = row[0]
                    decs[r] = row[1]
                    mags[r] = row[2]
        else:
            if verbose >= 1:
                print("submitting new query to Gaia: \n")
            jobstr = "SELECT TOP {} gaia_source.ra,gaia_source.dec,gaia_source.{} FROM gaiaedr3.gaia_source\n".format(nrefinepts, filtkey)
            jobstr += "WHERE 1=CONTAINS(POINT('ICRS', gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec),"
            jobstr += "CIRCLE('ICRS',{},{},{}))\n".format(ra,dec,fieldradius)
            jobstr += "ORDER by gaiaedr3.gaia_source.{} ASC".format(filtkey)
            if verbose >= 1:
                print(jobstr)
            job = Gaia.launch_job_async(jobstr, dump_to_file=True, output_file='{}\\gaiqueries\\gaiatable_ra{:.9}_dec{:.9}_{}_{}_{}_{}.csv'.format(user_dir,ra, dec, fieldquery, catalogue, filter, nrefinepts), output_format='csv')
            r = job.get_results()

            ras = np.array(r['ra'])
            decs = np.array(r['dec'])
            mags = np.array(r['{}'.format(filtkey)])
        
    elif catalogue == "SIMBAD":
        pass    ###update later

    if debug >= 1:
        print("Retrieved catalog Magnitudes:\n", mags)
        print("Retrieved catalog RAs:\n", ras)
        print("Retrieved catalog Decs:\n", decs)

    if debug >= 1:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits

        fig0 = plt.figure(0,figsize=(10,8))
        axes0 = fig0.add_subplot(111)
        axes0.invert_xaxis()
        axes0.axis('equal')
        axes0.scatter(ras,decs)
        plt.show()
        plt.close()

    catalogdata = np.stack((ras,decs,mags),axis=-1)
    catalogdata = catalogdata[~np.isnan(catalogdata).any(axis=1)]       ###remove nans

    rasdecsmags = np.vsplit(np.transpose(catalogdata),3)        ###splitting back into original form
    ras = rasdecsmags[0].flatten()
    decs = rasdecsmags[1].flatten()
    mags = rasdecsmags[2].flatten()

    num_catsources = ras.size              ###The number of catalog results returned by the Web Query minus the number of NaN rows removed

    rasum = 0
    for i in range(num_catsources):
        rasum += ras[i]
    a0 = rasum/num_catsources*pi/180

    
    decsum = 0
    for j in range(num_catsources):
        decsum += decs[j]
    d0 = decsum/num_catsources*pi/180

    meancatcoords[0] = a0
    meancatcoords[1] = d0

    # Gnomonic projection. See e.g. https://apps.dtic.mil/sti/pdfs/ADA037381.pdf, beginning of chapter 6 for a derivation of the equations. The context
    # in the paper is creating a map projection of the earth's surface. Note that the derived equations on page 208 have the scale incorporated into them.
    # The scale factor "S" is the dimensionless ratio of map distance/earth distance, and "a" is the radius of the earth. Thus the x and y end up in map 
    # distance. In this program, the scale factor has units of radians/pixel (originally it is supplied by the user in arcseconds/pixel, but this is converted).
    # However, the scale factor is not included in the intermediate coordinate equations, so the intermediate coordinates are left dimensionless, or in radians.
    # The conversion to "map distance", or pixels in this case, comes later in the main part of the WCS solver, which uses the scale factor in the transformations.
    # In the forward transformations from pixels to intermediate coordinates, for example, a dimensional analysis would read: pixels X radians/pixel = radians,
    # which are the correct units of the intermediate coordinates.
    #
    # Note the negative signs in the definition of both the X and Y intermediate coordinates. There is a different explanation for each of the sign changes.
    # 
    # The X sign flip occurs because the gnomonic projection assumes the observer is making a map looking down on the outside (convex) side of the earth's
    # surface, whereas in our case we are making a projection looking at the inside (concave) side of the celestial sphere. This change in persepective 
    # effectively puts a minus sign in the X coordinates.
    # 
    # The Y sign flip occurs because of a compensation we have to do to align the coordinates of the "true" projection with the coordinates of the FITS image.
    # Because the origin of the FITS image is at the top left corner, positive y increases as the pixels get visually lower in the image. On the other hand,
    # the gnomonic projection's Y axis points parallel to the celestial pole (corresponding with traditional "up"). Therefore we flip the projected Y coordinates
    # so that the sources are "aligned". In the case of no image rotation, the constellations of the PSE sources and the projected ("intermediate") sources would
    # then have the same orientation. Therefore, the rotation found by the WCS solver will be relative to this and be the true rotation, and not 180 degrees off.

    for k in range(num_catsources):
        a = ras[k]*pi/180
        d = decs[k]*pi/180
        X = (cos(d)*sin(a-a0) / (cos(d0)*cos(d)*cos(a-a0)+sin(d0)*sin(d)))
        Y = (cos(d0)*sin(d) - cos(d)*sin(d0)*cos(a-a0)) / (cos(d0)*cos(d)*cos(a-a0) + sin(d0)*sin(d))
        intrmcoords[k,0] = -X
        intrmcoords[k,1] = -Y
    
    if debug >= 1:
        print("(a0,d0) = ({},{})".format(a0,d0))

    if debug >= 1:
        print("Intermediate coordinates projected from catalog RAs and Decs:\n",np.asarray(intrmcoords))
    
    return num_catsources

def WCS(ra, dec, scale, scalebnds, rotation, rotationbnds, fieldradius, fieldquery, catalogue, filter, npts, nrefinepts, vertextol, smetadata, sindmap_initial, sindmap_refine, img_xmax, img_ymax, minmatches, kerneldiam, num_psesources, headervals, user_dir, filename, verbose, debug):
    ''' '''

    if debug >= 1:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from astropy.io import fits

    allintrmcoords = np.zeros((nrefinepts,2))
    meancatcoords = np.zeros(2)
    num_catsources = getIntermediateCoords(ra, dec, fieldradius, fieldquery, catalogue, filter, nrefinepts, allintrmcoords, meancatcoords, user_dir, verbose, debug)
    
    solution_initial = np.zeros(4)
    findInitialSolution(allintrmcoords, num_catsources, smetadata, num_psesources, sindmap_initial, npts, img_xmax, img_ymax, scale, scalebnds, rotation, rotationbnds, vertextol, minmatches, kerneldiam, solution_initial, user_dir, filename, verbose, debug)
    if verbose >= 1:
        print("initial solution ",solution_initial)

    solution_refined = np.zeros(6)
    refineSolution(allintrmcoords, num_catsources, smetadata, num_psesources, npts, nrefinepts, img_xmax, img_ymax, sindmap_refine, solution_initial[0], solution_initial[1], solution_initial[2], solution_initial[3], solution_refined, user_dir, filename, verbose, debug)
    if verbose >= 1:
        print("refined solution ",solution_refined)

    CD1_1 = solution_refined[0]*180/pi
    CD1_2 = solution_refined[1]*180/pi
    CD2_1 = solution_refined[2]*180/pi
    CD2_2 = solution_refined[3]*180/pi
    CRPIX1 = solution_refined[4]
    CRPIX2 = solution_refined[5]
    CRVAL1 = meancatcoords[0]*180/pi
    CRVAL2 = meancatcoords[1]*180/pi
    CDELT1 = sqrt(CD1_1*CD1_1+CD1_2*CD1_2)*3600
    CDELT2 = sqrt(CD2_1*CD2_1+CD2_2*CD2_2)*3600
    CROTA1 = atan2(CD1_2,-CD1_1)*180/pi
    CROTA2 = atan2(-CD2_1,-CD2_2)*180/pi

    headervals[0] = CRPIX1
    headervals[1] = CRPIX2
    headervals[2] = CRVAL1
    headervals[3] = CRVAL2
    headervals[4] = CD1_1
    headervals[5] = CD1_2
    headervals[6] = CD2_1
    headervals[7] = CD2_2
    headervals[8] = CDELT1
    headervals[9] = CDELT2
    headervals[10] = CROTA1
    headervals[11] = CROTA2