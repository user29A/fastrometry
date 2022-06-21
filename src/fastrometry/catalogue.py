from pathlib import Path
import csv
import numpy as np
from math import sin, cos, pi
from AstraCarta import astracarta
import sys, os

def getIntermediateCoords(ra, dec, scale, img_xmax, img_ymax, shape, filter, catalogue, pmepoch, nrefinepts, allintrmcoords, meancatcoords, user_dir, debug_report, verbosity, debug):

    if not Path('{}\\gaiaqueries'.format(user_dir)).is_dir():
        if verbosity >= 1:
            print("| Creating {}\\gaiaqueries".format(user_dir))
        Path('{}\\gaiaqueries'.format(user_dir)).mkdir(parents=True)

    if verbosity == 0:      ###Workaround for now
        console = sys.stdout    
        sys.stdout = open(os.devnull,'w')
    resultsfilename = astracarta(ra=ra, dec=dec, scale=scale, maglimit=30, pixwidth=img_xmax, pixheight=img_ymax, shape=shape, filter=filter, catalogue=catalogue, pmepoch=pmepoch, nquery=nrefinepts, outdir=user_dir+'gaiaqueries\\')
    if verbosity == 0:
        sys.stdout = console

    if resultsfilename == '':
        sys.exit("ERROR: Catalogue query failed to complete.")

    ras = np.array([])
    decs = np.array([])
    mags = np.array([])
    if filter == 'bp':
        filtername = 'phot_bp_mean_mag'
    elif filter == 'g':
        filtername = 'phot_g_mean_mag'
    elif filter == 'rp':
        filtername = 'phot_rp_mean_mag'

    with open(resultsfilename, 'r') as datafile:
        reader = csv.DictReader(datafile)
        for row in reader:
            x = float(row['ra'])
            y = float(row['dec'])
            z = float(row[filtername])
            ras = np.append(ras,x)
            decs = np.append(decs,y)
            mags = np.append(mags,y)

    if verbosity >= 1:
        print("| done")

    num_catsources = ras.size   ###"ras.size" is the number of catalog results returned by the Web Query minus the number of NaN rows removed
    if verbosity == 2:
        print("| Got {} valid sky coordinates.".format(num_catsources))

    if verbosity >= 1:
        print("| Gnomonically projecting sky coordinates...")

    rasum = 0
    for i in range(num_catsources):
        rasum += ras[i]
    a0 = rasum/num_catsources*pi/180
    a0deg = rasum/num_catsources

    decsum = 0
    for j in range(num_catsources):
        decsum += decs[j]
    d0 = decsum/num_catsources*pi/180
    d0deg = decsum/num_catsources

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

    for k in range(num_catsources):
        a = ras[k]*pi/180
        d = decs[k]*pi/180
        X = (cos(d)*sin(a-a0) / (cos(d0)*cos(d)*cos(a-a0)+sin(d0)*sin(d)))
        Y = (cos(d0)*sin(d) - cos(d)*sin(d0)*cos(a-a0)) / (cos(d0)*cos(d)*cos(a-a0) + sin(d0)*sin(d))
        allintrmcoords[k,0] = X
        allintrmcoords[k,1] = Y

    if verbosity >= 1:
        print("| done")
    
    if debug:
        np.savetxt(user_dir+"\\debug\\"+debug_report+"\\allintrmcoords.csv",allintrmcoords,delimiter=",")

    if debug:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14,8))
        plt.subplots_adjust(bottom=0.2)

        axes_query = fig.add_subplot(121)
        axes_query.invert_xaxis()
        axes_query.axis('equal')
        axes_query.set_title('Sky coordinates')
        axes_query.set_xlabel('Right ascension (degrees)')
        axes_query.set_ylabel('Declination (degrees)')
        axes_query.scatter(ras,decs,marker=".",color='red')
        axes_query.scatter([a0deg],[d0deg],marker="x",color='black')

        axes_proj = fig.add_subplot(122)
        axes_proj.invert_xaxis()
        axes_proj.axis('equal')
        axes_proj.set_title('Intermediate coordinates')
        axes_proj.set_xlabel('X (radians)')
        axes_proj.set_ylabel('Y (radians)')
        axes_proj.scatter(allintrmcoords[:,0],allintrmcoords[:,1],marker=".",color='red')
        axes_proj.scatter([0],[0],marker="x",color='black')
        
        dscrp_query = "Sky coordinates obtained from the catalogue. When the RA and Dec axes are \nscaled equally, the resulting shape will be elliptical, especially as \ndeclination approaches +- 90."
        dscrp_proj = "Intermediate coordinates, formed from taking the Gnomonic projection of the \ncatalog coordinates, using the mean of the coordinates as the projection \ncenter (invariant point)."
        plt.figtext(0.3, 0.05, dscrp_query, ha="center", fontsize=9)
        plt.figtext(0.72, 0.05, dscrp_proj, ha="center", fontsize=9)
        plt.savefig(user_dir+"\\debug\\"+debug_report+"\\projection.png")
        plt.show()
    
    return num_catsources