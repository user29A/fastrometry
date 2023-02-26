import csv
import numpy as np
from math import sin, cos, pi
from AstraCarta import astracarta
import sys

def getColumnData(resultsfile, catalogue_points, filter, verbosity):
    if filter == "g":   
        filtername = "phot_g_mean_mag"
    elif filter == "bp":
        filtername = "phot_bp_mean_mag"
    elif filter == "rp":
        filtername = "phot_rp_mean_mag"

    num_catsources = 0
    with open(resultsfile, 'r') as datafile:
        reader = csv.DictReader(datafile)
        for row in reader:
            catalogue_points[num_catsources,0] = float(row['ra'])*pi/180
            catalogue_points[num_catsources,1] = float(row['dec'])*pi/180
            catalogue_points[num_catsources,2] = float(row[filtername])
            num_catsources += 1
    return num_catsources

def gnomonicProject(catalogue_points, a0, d0, num_catsources, allintrmpoints, verbosity):
    for k in range(num_catsources):
        a = catalogue_points[k,0]
        d = catalogue_points[k,1]
        X = (cos(d)*sin(a-a0) / (cos(d0)*cos(d)*cos(a-a0)+sin(d0)*sin(d)))
        Y = (cos(d0)*sin(d) - cos(d)*sin(d0)*cos(a-a0)) / (cos(d0)*cos(d)*cos(a-a0) + sin(d0)*sin(d))
        allintrmpoints[k,0] = X
        allintrmpoints[k,1] = Y

def getIntermediateCoords(ra, dec, scale, img_xmax, img_ymax, shape, buffer, filter, catalogue, pmepoch, nrefinepts, allintrmpoints, catalogue_points, mean_catcoords, gaiaqueries, debug_report, overwrite, debug, verbosity):

    if verbosity == 0:
        silent = True
    else:
        silent = False

    resultsfile = astracarta(ra=ra, dec=dec, scale=scale, maglimit=30, pixwidth=img_xmax, pixheight=img_ymax, buffer=buffer, shape=shape, filter=filter, catalogue=catalogue, pmepoch=pmepoch, nquery=nrefinepts, outdir=gaiaqueries, silent=silent, overwrite=overwrite)
    if resultsfile == '':
        sys.exit("ERROR: Catalogue query failed to complete.")

    num_catsources = getColumnData(resultsfile, catalogue_points, filter, verbosity)
    if verbosity == 2:
        print("| Got {} valid sky coordinates from the catalogue.".format(num_catsources))

    a0 = np.mean(catalogue_points[:num_catsources,0])
    d0 = np.mean(catalogue_points[:num_catsources,1])

    mean_catcoords[0] = a0
    mean_catcoords[1] = d0

    gnomonicProject(catalogue_points, a0, d0, num_catsources, allintrmpoints, verbosity)

    if debug:
        np.savetxt(debug_report/"catalogue_points.csv", catalogue_points, delimiter=",")
        np.savetxt(debug_report/"allintrmpoints.csv", allintrmpoints, delimiter=",")

    if debug:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(13,7))

        axes_query = fig.add_subplot(121)
        axes_query.invert_xaxis()
        axes_query.axis('equal')
        axes_query.set_title('Sky coordinates')
        axes_query.set_xlabel('Right ascension (degrees)')
        axes_query.set_ylabel('Declination (degrees)')
        axes_query.scatter(catalogue_points[:num_catsources,0]*180/pi, catalogue_points[:num_catsources,1]*180/pi, marker=".", color='red')
        axes_query.scatter([a0*180/pi], [d0*180/pi], marker="x", color='black')

        axes_proj = fig.add_subplot(122)
        axes_proj.invert_xaxis()
        axes_proj.axis('equal')
        axes_proj.set_title('Intermediate coordinates')
        axes_proj.set_xlabel('X (radians)')
        axes_proj.set_ylabel('Y (radians)')
        axes_proj.scatter(allintrmpoints[:num_catsources,0], allintrmpoints[:num_catsources,1], marker=".",color='red')
        axes_proj.scatter([0], [0], marker="x", color='black')
        
        dscrp_query = "Sky coordinates obtained from the catalogue. When the RA and Dec axes are \nscaled equally, the resulting shape will be wider than it is tall, especially \nas declination approaches +- 90."
        dscrp_proj = "Intermediate coordinates, formed from taking the Gnomonic projection of the \ncatalog coordinates, using the mean of the coordinates as the projection \ncenter (invariant point)."
        plt.figtext(0.3, 0.08, dscrp_query, ha="center", fontsize=9)
        plt.figtext(0.72, 0.08, dscrp_proj, ha="center", fontsize=9)
        plt.subplots_adjust(wspace=0.2, left=0.08, right=0.92, bottom=0.25, top=0.9)
        plt.savefig(debug_report/"projection.png")
        plt.show()

    return num_catsources