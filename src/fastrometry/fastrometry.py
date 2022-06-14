#This code is always intended to be imported as a module, whether via the fastrometry command-line call (which uses the entry_points
#option in setup.cfg which implicitly imports this module) or via a specified import statement in another (parent) module. Thus
#calling this module directly will fail because of the relative imports used in the next two lines.

from .cython_code import PSE
from .cython_code import WCS

import argparse
from astropy.io import fits
import numpy as np
from pathlib import Path
import re
import json
from textwrap import wrap
from math import pi, sqrt

################
VERSION="0.0.1"
################

def insertCopyNumber(outfilename, filename):
    copynum = 1
    while Path(outfilename).is_file():
        outfilename = filename.split('.')[0]+' WCS ({}).fits'.format(copynum)
        copynum += 1
    return outfilename

def formatting(text): #Thank you to user blackpen on StackExchange for sharing this function
    text=re.sub('\s+',' ',text); text=re.sub('^\s+','',text); text=re.sub('\s+$','',text)
    text=wrap(text,width=80,initial_indent=' '*4,subsequent_indent=' '*4)
    s=""
    for i in (text): s=s+i+"\n"
    s=re.sub('\s+$','',s)
    return(s+'\n \n')

def getArgsFromCommandLine():
    intro_paragraph = 'To use, type "fastrometry", followed by the 4 required options (specified below) and any number of other options. Make sure your current working directory is the one containing the FITS images you wish to process.\n\n=======================================================================\n| For a description of how the package works and what it does, please |\n| visit https://github.com/user29A/fastrometry/wiki.                  |\n=======================================================================\n\nCredits:\n\nWritten by Cameron Leahy under the supervision of Joseph Postma. Based on the C# code in JPFITS written by Joseph Postma.\n\nThis project was undertaken with the financial support of the Canadian Space Agency.'
    indent_formatter = lambda prog: argparse.RawTextHelpFormatter(prog,max_help_position=40)
    parser = argparse.ArgumentParser(description='Astronomy package written in Python\n\n{}\n\nRequired options are:\n\n-filename\n-scale\n-ra\n-dec'.format(intro_paragraph),add_help=False,formatter_class=indent_formatter)
    parser.add_argument('-h', '-help', '--h', '--help', action='help', default=argparse.SUPPRESS, help=formatting('Shows this help message.'))
    parser.add_argument('-filename', type=Path, help=formatting('This is the full file path + name of a FITS image file to solve the WCS solution for.'))
    parser.add_argument('-ra', help=formatting('The approximate right-ascension of the field center of the image. This can be supplied in either right-ascension sexagesimal (HH:MM:SS.S) format, degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either right-ascension sexagesimal or degree.decimal format.'))
    parser.add_argument('-dec', help=formatting('The approximate declination of the field center of the image. This can be supplied in either declination sexagesimal format (DD:MM:SS.S), degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either declination sexagesimal or degree.decimal format.'))
    parser.add_argument('-scale', help=formatting('This is the approximate field scale, in arcseconds per pixel.'))
    parser.add_argument('-scalebnds', help=formatting('This is the "plus or minus" range of the field scale, in the same units as the field scale of arcseconds per pixel. If no scalebnds are supplied then a +-5%% range bound is assumed. Zero is a valid option if the scale is known precisely, and overrides the +-5%% assumption default, and will increase solve speed.'))
    parser.add_argument('-rotation', help=formatting('Use this to provide an initial estimate of the image field rotation relative to sky coordinates, between +- 180 degrees. Units in degrees. Zero degrees corresponds to the CAST convention, as does positive angle measure, and negative angle is opposite rotation to that. If not supplied then the solver automatically estimates the rotation and also its upper and lower estimate bounds; the auto-estimator mitigates the need for the user to provide a rotation estimate, but the user may still do so.'))
    parser.add_argument('-rotationbnds', help=formatting('This is the "plus or minus" range bounds of the rotation estimate. Zero is a legitimate value if the rotation is known precisely. If passed without setting -rotation, the rotation will be assumed to be zero, and -rotationbnds will be the +- bound around 0 degrees. Units in degrees. The smaller the range, the faster the solve speed. If not supplied then the solver automatically estimates the rotation and also its upper and lower estimate bounds; the auto-estimator mitigates the need for the user to provide a rotation estimate, but the user may still do so. '))
    parser.add_argument('-fieldradius', help=formatting('The field radius is nominally calculated from the supplied image dimensions, using the larger of the NAXIS1, NAXIS2 keywords if they are not equal, and the nominal field scale. If the user wishes to specify a radius, for example if there is significant image padding, then use this option. Note that this is the radius for a circle. Units in arcminutes.'))
    parser.add_argument('-fieldquery', help=formatting('Options are "inside" or "outside", with default "inside". The only meaningful catalogue query via astroquery or gaiaquery is a circle query, since typically the WCS will not yet exist in order to calculate the corners of a square image to issue a POLYGON query, and there may be arbitrary image rotation, etc. Thus, an "inside" query requests catalogue sources given a circle with tangents intersecting the sides of a square image, whereas an "outside" query requests sources given a circle with tangents intersecting the corners of a square image. The square image in either case is given by the larger of the NAXIS1/2 keywords. A circular aperture with no image padding will be correspondent with the "inside" query, whereas a square image may have sources in the corners which the catalogue query would not retrieve in this case. An "outside" query for a circular image will retrieve catalogue sources from a larger radius (sqrt(2) times larger) than the image circle, whereas for a square image the retrieval will fill the entire image; in both cases there will be extraneous catalogue sources, either at distances larger than the nominal field radius, or outside of the edges of the square.'))
    parser.add_argument('-catalogue', help=formatting('This specifies the catalogue to query for sources in the given nominal image field. The default is Gaia DR3, in its \'g\' band. To specify other filter bands, see the -filter option setting. Possible catalogues options are: \n\n \'GaiaDR3\'.'))
    parser.add_argument('-filter', help=formatting('This specifies the filter to use from a given catalogue. Default catalogue and filter are Gaia DR3 in \'g\' band. If a specified filter isn\'t found in a given catalogue (specified or default), an error is thrown. Valid filters for given catalogues are as follows: \n\nGaia DR3: \'bp\' (blue), \'g\' (green), \'rp\' (red).'))
    parser.add_argument('-npts', help=formatting('The number of sources to use in both the catalogue and image source extraction. Suggest 25 for good wavelength correspondence between the catalogue and image; 50 for poor correspondence; 100 for very poor; 200 for a hope and a prayer and you might crash your system...but it still works often. 50 is the default when not supplied.'))
    parser.add_argument('-nrefinepts', help=formatting('This is the number of points to use for automatic refinement of the initial solution based on npts; npts must be a relatively smaller value (see comment on npts) given the (n-choose-3)^2 nature of the solver, but the initial solution may be easily refined with a much larger number of points. Default is 500; maximum is 1000; minimum is npts.'))
    parser.add_argument('-pixsat', help=formatting('This is important and very helpful to use if the image is known to have saturated pixels. One can inform the point source extractor the threshold above which sources will be expected to be saturated and form island-plateaus instead of Gaussian peaks. For example, a pixel-well may saturate at an ADU value of ~50,000, although it might have a 16-bit ADC which would saturate at 65,536 pixel value. In this case a \'pixsat\' should be specified as, say, 45,000, just below the ADU well-depth. If -pixsat is not supplied, there will be no special treatment of saturated sources.'))
    parser.add_argument('-kernelrad', help=formatting('This is the radius of the centroiding kernel used in the point-source extractor. Default, and minimum, is 2, which means that +-2 pixels about a source peak will be used for centroiding the source, i.e., a 5x5 pixel array centered on the source peak is centroided. If the sources are over-sampled then the user might like to set a larger radius such that the kernel samples the complete PSF. The default of 2 assumes that the image source PSFs are critically-sampled.'))
    parser.add_argument('-sourcesep', help=formatting('This sets the minimum separation allowed between sources the image, in unit pixels. Default is 25. If it is less than the centroid kernel radius, it is made equal to the centroid kernel radius.'))
    parser.add_argument('-vertextol', help=formatting('Default 0.25 degrees. This sets the vertex tolerance within which two vertices of a triangle must match in angle to another triangle in order to be considered a potential pattern-match between the image source coordinates and catalogue source coordinates. Not much testing has been done on this option and 0.25 has always been used. A smaller tolerance will be less forgiving to distortion but would offer faster searching and fewer false-positives, whereas a larger tolerance will be more forgiving to distortion at the expense of slower searching and a higher rate of false-positives.'))
    parser.add_argument('-nmatchpoints', help=formatting('This specifies the number of points which must match between the point source list and catalogue list to be considered a solution. Default is 6 points. Minimum is 3 but this is very likely to produce false-positive solutions. 4 or 5 can be used where there are simply very few points in the source image. 6 or higher should guarantee no false positives.'))
    parser.add_argument('-nmatchpercent', help=formatting('Same as above but as a percentage of point sources available. Default is 25%%. If both nmatchpoints and nmatchpercent are supplied, the smaller threshold will be used, found by comparing nmatchpoints to nmatchpercent*npts.'))
    parser.add_argument('--save', metavar='ID', help=formatting('Saves current options to a json file. The user must supply an ID consisting of arbitrary characters that will get written into the filename. This way, multiple sets of options can be saved.'))
    parser.add_argument('--load', metavar='ID', help=formatting('Loads options from the json file whose filename contains the ID specified. Any supplied options will be overwritten by the ones in the json file.'))
    parser.add_argument('--version', '-V', action='version', version='fastrometry {}'.format(VERSION), help=formatting('Shows the current version of fastrometry.'))
    parser.add_argument('--verbose', nargs='?', default=0, const=1, help=formatting('Intended for normal (non-debugging) use. Sets the verbosity level of console messages. If not present, this option defaults to 0, or minimal output. If provided without a value, it assumes a value of 1, or moderate verbosity. Set to 2 for maximum verbosity.'))
    parser.add_argument('--debug', nargs='?', default=0, const=1, help=formatting('Enters debug mode. Steps through the program, pausing at intervals to print diagnostic and debugging info to the console and display various plots and images. If not present, this option defaults to 0, or non-debug mode. If provided without a value, it assumes a value of 1, or normal debug mode. For the maximum amount of diagnostic info, set debug to 2. In both cases a debug report will be created and saved to the current working directory. --verbose is automatically set to maximum.'))

    input_args = parser.parse_args()

    return input_args

def find_WCS(filename=None, ra=None, dec=None, scale=None, scalebnds=None, rotation=None, rotationbnds=None, fieldradius=None, fieldquery=None, catalogue=None, filter=None, npts=None, nrefinepts=None, pixsat=None, kernelrad=None, sourcesep=None, vertextol=None, nmatchpoints=None, nmatchpercent=None, save=None, load=None, verbose=None, debug=None):
    
    user_dir = str(Path.cwd())

    try:
        verbose = int(verbose)
        if 0 <= verbose <= 2:
            pass
        else:
            print('ERROR: verbose must be between 0 and 2.')
            exit()
    except:
        print('ERROR: verbose must be an integer.')
        exit()

    try:
        debug = int(debug)
        if 0 <= debug <= 2:
            if debug == 0:
                pass
            if debug == 1:
                verbose = 2
            if debug == 2:
                verbose = 2
        else:
            print('ERROR: debug must be between 0 and 2.')
            exit()
    except:
        print('ERROR: debug must be an integer.')
        exit()

    if load is None:
        if filename is None:
            print('ERROR: -filename is required.')
            exit()
        else:
            try:
                filename = str(filename)
                filepath = user_dir+'\\'+filename
                hdul = fits.open('{}'.format(filepath))
                img = hdul[0].data.astype(np.double)
                header = hdul[0].header
            except Exception as e:
                print('ERROR: Could not open the specified FITS file. Check that you are in the directory containing your FITS files.\nORIGINAL ERROR: {}'.format(e))
                exit()

        if ra is None:
            print('ERROR: -ra is required.')
            exit()
        else:
            if len(ra.split(':')) == 1:
                try:
                    ra = float(ra)              ###decimal
                    if 0 <= ra < 360:
                        pass
                    else:
                        print('ERROR: ra in degree.decimal format must be between 0 and 360.')
                        exit()
                except:                            
                    if ra in header:       ###header keywords
                        ra = float(header['{}'.format(ra)])
                    else:
                        print('ERROR: Could not find the specified keyword in the FITS header (assuming ra was given as a keyword).')
                        exit()
            elif len(ra.split(':')) == 3:      ###sexagesimal
                ra_hms = ra.split(':')
                try:
                    hrs = int(ra_hms[0])
                    mins = int(ra_hms[1])
                    secs = float(ra_hms[2])
                    if 0 <= hrs < 24 and  0 <= mins < 60 and 0 <= secs < 60:
                        ra = (hrs + mins/60 + secs/3600)*15
                    else:
                        print('ERROR: ra in HH:MM:SS.S format must hrs between 0 and 24, minutes between 0 and 60, and seconds between 0 and 60)')
                        exit()
                except:
                    print('ERROR: ra in HH:MM:SS.S format must have integer hours, integer minutes, and integer or decimal seconds.')
                    exit()
            else:
                print('ERROR: ra must be entered either in HH:MM:SS.S format (e.g. "18:44:27.891"), degree.decimal format (e.g. "78.132"), or as a FITS header keyword whose value will be retrieved (e.g. "RA_PT").')
                exit()

        if dec is None:
            print('ERROR: -dec is required.')
            exit()
        else:
            if len(dec.split(':')) == 1:
                try:
                    dec = float(dec)                 ###decimal
                    if -90 <= dec < 90:
                        pass
                    else:
                        print('ERROR: dec in degree.decimal format must be between -90 and 90.')
                        exit()
                except:
                    if dec in header:        ###header keywords
                        dec = float(header['{}'.format(dec)])
                    else:
                        print('ERROR: Could not find the specified keyword in the FITS header (assuming dec was given as a keyword).')
                        exit()
            elif len(dec.split(':')) == 3:     ###sexagesimal
                dec_hms = dec.split(':')
                try:
                    degs = int(dec_hms[0])
                    amins = int(dec_hms[1])
                    asecs = float(dec_hms[2])
                    if -90 <= degs < 90 and 0 <= amins < 60 and 0 <= asecs < 60:
                        dec = degs + amins/60 + asecs/3600
                    else:
                        print('ERROR: dec in DD:MM:SS.S" format must have degrees between -90 and 90, arcminutes between 0 and 60, and arcseconds between 0 and 60.')
                        exit()
                except:
                    print('ERROR: dec in DD:MM:SS.S" format must have integer degrees, integer arcminutes, and integer or decimal arcseconds.')
                    exit()
            else:
                print('ERROR: dec must be entered either in DD:MM:SS.S" format (e.g. "-49:12:09.255"), degree.decimal format (e.g. "78.132"), or as a FITS header keyword whose value will be retrieved (e.g. "DEC_PT").')
                exit()

        if scale is None:
            print('ERROR: scale is required.')
            exit()
        else:
            try:
                scale = float(scale)
                if scale > 0:
                    pass
                else:
                    print('ERROR: scale must be greater than 0.')
                    exit()
            except:
                print('ERROR: scale must be a number.')
                exit()

        if scalebnds is None:
            scalebnds = 5.
        else:
            try:
                scalebnds = float(scalebnds)
                if scalebnds >= 0:
                    pass
                else:
                    print('ERROR: scalebnds must be greater than 0.')
                    exit()
            except:
                    print('ERROR: scalebnds must be a number.')
                    exit()

        if rotation is None and rotationbnds is None:
            rotation = None       #auto
            rotationbnds = None        #auto
        elif rotation is None and rotationbnds is not None:
            try:
                rotationbnds = float(rotationbnds)
                if 0 <= rotationbnds <= 180:
                    rotation = None        #auto
                else:
                    print('ERROR: rotationbnds must be between 0 and 180.')
                    exit()
            except:
                print('ERROR: rotationbnds must be a number.')
                exit()
        elif rotation is not None and rotationbnds is None:
            try:
                rotation = float(rotation)
                if -180 <= rotation < 180:
                    rotationbnds = None       #auto
                else:
                    print('ERROR: rotation must be between -180 and 180.')
                    exit()
            except:
                print('ERROR: rotation must be a number.')
                exit()
        elif rotation is not None and rotationbnds is not None:
            try:
                rotation = float(rotation)
                if -180 <= rotation < 180:
                    pass
                else:
                    print('ERROR: rotation must be between -180 and 180.')
                    exit()
            except:
                print('ERROR: rotation must be a number.')
                exit()
            try:
                rotationbnds = float(rotationbnds)
                if 0 <= rotationbnds <= 180:
                    pass
                else:
                    print('ERROR: rotationbnds must be between 0 and 180.')
                    exit()
            except:
                print('ERROR: rotationbnds must be a number')
                exit()

        if fieldradius is None and fieldquery is None:
            fieldquery = 'inside'
            if header['NAXIS1'] >= header['NAXIS2']:
                fieldradius = header['NAXIS1']/2*scale/60
            elif header['NAXIS2'] > header['NAXIS1']:
                fieldradius = header['NAXIS2']/2*scale/60
        elif fieldradius is None and fieldquery is not None:
            if fieldquery in ['inside','outside']:
                fieldquery = fieldquery
                if header['NAXIS1'] >= header['NAXIS2']:
                    fieldradius = header['NAXIS1']/2*scale/60
                elif header['NAXIS2'] > header['NAXIS1']:
                    fieldradius = header['NAXIS2']/2*scale/60
                if fieldquery == "outside":
                    fieldradius = fieldradius*sqrt(2)
            else:
                print("ERROR: fieldquery must be from the following options: ['inside','outside'].")
                exit()
        elif fieldradius is not None and fieldquery is None:
            try:
                fieldradius = float(fieldradius)
                fieldquery = 'inside'
            except:
                print('ERROR: fieldradius must be a number.')
                exit()
        elif fieldradius is not None and fieldquery is not None:
            print("ERROR: fieldradius and fieldquery cannot both be specified, since the fieldradius is calculated from the argument of fieldquery, if present.")
            exit()

        if catalogue is None:
            catalogue = 'GaiaDR3'
            if filter is None:
                filter = 'g'
            elif filter in ['bp','g','rp']:
                pass
            else:
                print("ERROR: filter for the GaiaDR3 catalog must be from the following options: ['bp','g','rp'].")
                exit()
        elif catalogue in ['GaiaDR3','Spitzer']:
            catalogue = catalogue
            if catalogue == 'GaiaDR3':
                if filter is None:
                    filter = 'g'
                elif filter in ['bp','g','rp']:
                    pass
                else:
                    print("ERROR: filter for the GaiaDR3 catalog must be from the following options: ['bp','g','rp'].")
                    exit()
            elif catalogue == 'SIMBAD':
                filter = 'unknown'          ###work on in a future update
        else:
            print("ERROR: catalogue must be from the following options: ['GaiaDR3','Spitzer'].")
            exit()

        if npts is None:
            npts = 50
        else:
            try:
                npts = int(npts)
                if npts >= 10:
                    if npts > 200:
                        from tkinter import messagebox
                        messagebox.showwarning('Warning: computer may explode.')
                else:
                    print("ERROR: npts must be greater than or equal to 10.")
                    exit()
            except:
                print('ERROR: npts must be an integer.')
                exit()

        if nrefinepts is None:
            nrefinepts = 500
        else:
            try:
                nrefinepts = int(nrefinepts)
                if npts <= nrefinepts <= 1000:
                    pass
                else:
                    print("ERROR: nrefinepts must be between npts and 1000.")
                    exit()
            except:
                print('ERROR: nrefinepts must be an integer.')
                exit()

        if pixsat is None:
            pixsat = 0.0
        else:
            try:
                pixsat = float(pixsat)
                if pixsat >= 0: 
                    pass
                else:
                    print("ERROR: pixsat must be greater than or equal to 0.")
                    exit()
            except:
                print('ERROR: pixsat must be a number.')
                exit()

        if kernelrad is None:
            kernelrad = 2
        else:
            try:
                kernelrad = int(kernelrad)
                if kernelrad > 2:
                    pass
                else:
                    print("ERROR: kernelrad must be greater than or equal to 2.")
                    exit()
            except:
                print("ERROR: kernelrad must be an integer.")
                exit()

        if sourcesep is None:
            sourcesep = 25
        else:
            try:
                sourcesep = int(sourcesep)
                if sourcesep >= kernelrad:
                    pass
                else:
                    print("ERROR: sourcesep must be greater than or equal to the supplied kernelrad.")
                    exit()
            except:
                print("ERROR: sourcesep must be an integer.")
                exit()

        if vertextol is None:
            vertextol = 0.25
        else:
            try:
                vertextol = float(vertextol)
                if 0 < vertextol < 2:
                    pass
                else:
                    print("ERROR: vertextol must be between 0 and 2.")
                    exit()
            except:
                print("ERROR: vertextol must be a number.")
                exit()

        if nmatchpoints is None:
            nmatchpoints = 6
        else:
            try:
                nmatchpoints = int(nmatchpoints)
                if nmatchpoints > 2:
                    pass
                else:
                    print("ERROR: nmatchpoints must be greater than 2.")
                    exit()
            except:
                print("ERROR: nmatchpoints must be an integer.")
                exit()

        if nmatchpercent is None:
            nmatchpercent = 25
        else:
            try:
                nmatchpercent = float(nmatchpercent)
                if nmatchpercent > 10:
                    pass
                else:
                    print("ERROR: nmatchpercent must be greater than 10.")
                    exit()
            except:
                print("ERROR: nmatchpercent must be a number.")
                exit()

        if save is None:
            pass
        else:
            try:
                contents = {
                    'filename' : filename,
                    'ra' : ra,
                    'dec' : dec,
                    'scale' : scale,
                    'scalebnds' : scalebnds,
                    'rotation' : rotation,
                    'rotationbnds' : rotationbnds,
                    'fieldradius' : fieldradius,
                    'fieldquery' : fieldquery,
                    'catalogue' : catalogue,
                    'filter' : filter,
                    'npts' : npts,
                    'nrefinepts' : nrefinepts,
                    'pixsat' : pixsat,
                    'kernelrad' : kernelrad,
                    'sourcesep' : sourcesep,
                    'vertextol' : vertextol,
                    'nmatchpoints' : nmatchpoints,
                    'nmatchpercent' : nmatchpercent,
                }
                id = save
                jobject = json.dumps(contents)
                re.sub(r'[^\w\-_\. ]', '', id)
                if not Path('{}\\cmd_args'.format(user_dir)).is_dir():
                    if verbose >= 1:
                        print("Creating {}\\cmd_args".format(user_dir))
                    Path('{}\\cmd_args'.format(user_dir)).mkdir(parents=True)
                with open('{}\\cmd_args\\saveoptions_{}.json'.format(user_dir,id),'w') as f:
                    f.write(jobject)
                if verbose >= 1:
                    print("Save successful. ID: {}".format(id))
            except Exception as e:
                print(e)
                print("ERROR: Could not save options.")
                exit()
    else:
        try:
            with open('{}\\cmd_args\\saveoptions_{}.json'.format(user_dir,id),'r') as f:
                jobject = json.load(f)
                filename = jobject['filename']
                ra = jobject['ra']
                dec = jobject['dec']
                scale = jobject['scale']
                scalebnds = jobject['scalebnds']
                rotation = jobject['rotation']
                rotationbnds = jobject['rotationbnds']
                fieldradius = jobject['fieldradius']
                fieldquery = jobject['fieldquery']
                catalogue = jobject['catalogue']
                filter = jobject['filter']
                npts = jobject['npts']
                nrefinepts = jobject['nrefinepts']
                pixsat = jobject['pixsat']
                kernelrad = jobject['kernelrad']
                sourcesep = jobject['sourcesep']
                vertextol = jobject['vertextol']
                nmatchpoints = jobject['nmatchpoints']
                nmatchpercent = jobject['nmatchpercent']
                
                filepath = user_dir+'\\'+filename
                hdul = fits.open('{}'.format(filepath))
                img = hdul[0].data.astype(np.double)
                header = hdul[0].header
        except Exception as e:
            print(e)
            print("ERROR: Could not load options. Check that a json file exists with the specified id.")
            exit()


    #################### GLOSSARY OF IMPORTANT VARIABLE NAMES ####################
    ###
    ###     img:                The FITS image supplied by the user. The PSE function extracts the point sources in the image and puts their metadata (namely, coordinates and brightness) into smetadata ("source metadata").
    ###     sindmap:            The source index map. This is a copy of the image with every point set to a value of -1 initially. In the PSE function, circular areas (the kernel) around each found source are given
    ###                         non-zero values. This provides a way of checking if an inverse-transformed point in the WCS function "lands" on a source.
    ###     smetadata:          The source metadata array, which stores the coordinates, brightness, and background of every found source (up to 500). The background is only useful in the PSE calculations.              
    ###     psecoords_view:     An array holding the pixel coordinates of the <npts> brightest sources from the smetadata. The "view" suffix indicates that it is a "memory view", a type of high-efficency array
    ###                         pointer datatype in Cython.
    ###     intrmcoords_view:   An array holding the projected coordinates ("intermediate coordinates") of a set of sources downloaded from the catalogue. The success of the WCS is contingent on the correspondence
    ###                         between the intermediate coordinates and the pse coordinates.
    ###     S                   The scale parameter found by the optimizer in the WCS.
    ###     phi                 The rotation parameter found by the optimizer in the WCS.
    ###     CRPIXx              The reference pixel location's x coordinate, the third parameter found by the optimizer in the WCS. CRPIX is the point in the image distinguished by the property that vectors drawn
    ###                         from it to the psecoords only need to undergo a rotation and a scaling (no translation) in order to point to the corresponding intermediate coords.
    ###     CRPIXy              The reference pixel location's y coordinate, the fourth and final parameter found by the optimizer in the WCS.
    ###
    #####################################################################

    if rotation is not None:
        rotation = rotation*pi/180
    if rotationbnds is not None:
        rotationbnds = rotationbnds*pi/180
    img_xmax = int(img.shape[1])
    img_ymax = int(img.shape[0])
    img_xcenter = (img_xmax+1)/2
    img_ycenter = (img_ymax+1)/2
    if debug >= 1:
        print("fieldquery ",fieldquery)
        print("fieldradius (in m') ",fieldradius)
    fieldradius = fieldradius/60
    if fieldquery == "inside":
        for x in range(img_xmax):
            for y in range(img_ymax):
                if (x-img_xcenter)*(x-img_xcenter)/(img_xmax/2)**2+(y-img_ycenter)*(y-img_ycenter)/(img_ymax/2)**2 > 1:
                    img[y-1,x-1] = 0
    scale = scale/3600*pi/180
    scalebnds = .01*scalebnds
    vertextol = vertextol*pi/180
    sindmap_initial = -1*np.ones(img.shape,dtype=np.int32)
    sindmap_refine = -1*np.ones(img.shape,dtype=np.int32)
    smetadata = np.zeros((nrefinepts,3),dtype=np.double)
    
    num_psesources = PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, sindmap_initial, sindmap_refine, smetadata, verbose, debug) ### populate smetadata and sindmap

    if debug >= 1:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        unique, counts = np.unique(sindmap_initial, return_counts = True)
        res = np.column_stack((unique,counts))
        print(res)
        unique, counts = np.unique(sindmap_refine, return_counts = True)
        res = np.column_stack((unique,counts))
        print(res)
        xs = []
        ys = []
        for g in range(num_psesources):
            xs.append(smetadata[g,0])
            ys.append(smetadata[g,1])
        fig = plt.figure(3,figsize=(10,8))
        axes = fig.add_subplot(111)
        image_data = fits.getdata('{}\\{}}'.format(user_dir,filename))
        axes.imshow(image_data, cmap="gray", norm=LogNorm())
        axes.scatter(xs,ys,color='red',marker='.')
        plt.show()

    if nmatchpoints < nmatchpercent*npts:
        minmatches = nmatchpoints
    else:
        minmatches = nmatchpercent*npts
    kerneldiam = kernelrad*2+1

    headervals = np.zeros(12)
    WCS(ra, dec, scale, scalebnds, rotation, rotationbnds, fieldradius, fieldquery, catalogue, filter, npts, nrefinepts, vertextol, smetadata, sindmap_initial, sindmap_refine, img_xmax, img_ymax, minmatches, kerneldiam, num_psesources, headervals, user_dir, filename, verbose, debug)

    header['CTYPE1'] = ('RA--TAN', 'WCS type of horizontal coord. transformation')
    header['CTYPE2'] = ('DEC--TAN', 'WCS type of vertical coord. transformation')
    header['CRPIX1'] = (headervals[0], 'WCS coordinate reference pixel on axis 1')
    header['CRPIX2'] = (headervals[1], 'WCS coordinate reference pixel on axis 2')
    header['CRVAL1'] = (headervals[2], 'WCS coordinate reference value on axis 1 (deg)')
    header['CRVAL2'] = (headervals[3], 'WCS coordinate reference value on axis 2 (deg)')
    header['CD1_1'] = (headervals[4], 'WCS rotation and scaling matrix')
    header['CD1_2'] = (headervals[5], 'WCS rotation and scaling matrix')
    header['CD2_1'] = (headervals[6], 'WCS rotation and scaling matrix')
    header['CD2_2'] = (headervals[7], 'WCS rotation and scaling matrix')
    header['CDELT1'] = (headervals[8], 'WCS plate scale on axis 1 (arcsec per pixel)')
    header['CDELT2'] = (headervals[9], 'WCS plate scale on axis 2 (arcsec per pixel)')
    header['CROTA1'] = (headervals[10], 'WCS field rotation angle on axis 1 (degrees)')
    header['CROTA2'] = (headervals[11], 'WCS field rotation angle on axis 2 (degrees)')
    
    outfilename = filename.split('.')[0]+' WCS.'+filename.split('.')[1]
    numbered_outfilename = insertCopyNumber(outfilename,filename)
    outfilepath = user_dir+'\\'+numbered_outfilename
    if verbose >= 1:
        print("writing to {}".format(numbered_outfilename))
    fits.writeto('{}'.format(outfilepath), img, header, overwrite=True, output_verify='silentfix')

    print("done")

    return

def fastrometryCall():
    args = getArgsFromCommandLine()
    solution = find_WCS(filename=args.filename, ra=args.ra, dec=args.dec, scale=args.scale, scalebnds=args.scalebnds, rotation=args.rotation, rotationbnds=args.rotationbnds, fieldradius=args.fieldradius, fieldquery=args.fieldquery, catalogue=args.catalogue, filter=args.filter, npts=args.npts, nrefinepts=args.nrefinepts, pixsat=args.pixsat, kernelrad=args.kernelrad, sourcesep=args.sourcesep, vertextol=args.vertextol, nmatchpoints=args.nmatchpoints, nmatchpercent=args.nmatchpercent, save=args.save, load=args.load, verbose=args.verbose, debug=args.debug)

    