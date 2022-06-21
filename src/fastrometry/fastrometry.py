#This code is always intended to be imported as a module, whether via the fastrometry command-line call (which uses the entry_points
#option in setup.cfg which implicitly imports this module) or via a specified import statement in another (parent) module. Thus
#calling this module directly will fail because of the relative imports used in the next two lines.

from .cython_code import PSE
from .cython_code import WCS
from .catalogue import getIntermediateCoords

import argparse
from astropy.io import fits
from datetime import datetime
import numpy as np
from pathlib import Path
import re
import json
from textwrap import wrap
from math import pi
import sys

################
VERSION="1.0.0"
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
    indent_formatter = lambda prog: argparse.RawTextHelpFormatter(prog,max_help_position=40)
    parser = argparse.ArgumentParser(description='Astronomy package written in Python.\n\nRequired options are:\n\n-filename\n-scale\n-ra\n-dec',add_help=False,formatter_class=indent_formatter)
    parser.add_argument('-h', '-help', '--h', '--help', action='help', default=argparse.SUPPRESS, help=formatting('Shows this help message.'))
    parser.add_argument('-filename', type=Path, help=formatting('This is the full file path + name of a FITS image file to solve the WCS solution for.'))
    parser.add_argument('-ra', help=formatting('The approximate right-ascension of the field center of the image. This can be supplied in either right-ascension sexagesimal (HH:MM:SS.S) format, degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either right-ascension sexagesimal or degree.decimal format.'))
    parser.add_argument('-dec', help=formatting('The approximate declination of the field center of the image. This can be supplied in either declination sexagesimal format (DD:MM:SS.S), degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either declination sexagesimal or degree.decimal format.'))
    parser.add_argument('-scale', help=formatting('This is the approximate field scale, in arcseconds per pixel.'))
    parser.add_argument('-scalebnds', help=formatting('This is the "plus or minus" range of the field scale, in the same units as the field scale of arcseconds per pixel. If no scalebnds are supplied then a +-5%% range bound is assumed. Zero is a valid option if the scale is known precisely, and overrides the +-5%% assumption default, and will increase solve speed.'))
    parser.add_argument('-rotation', help=formatting('Use this to provide an initial estimate of the image field rotation relative to sky coordinates, between +- 180 degrees. Units in degrees. Zero degrees corresponds to the CAST convention, as does positive angle measure, and negative angle is opposite rotation to that. If not supplied then the solver automatically estimates the rotation and also its upper and lower estimate bounds; the auto-estimator mitigates the need for the user to provide a rotation estimate, but the user may still do so.'))
    parser.add_argument('-rotationbnds', help=formatting('This is the "plus or minus" range bounds of the rotation estimate. Zero is a legitimate value if the rotation is known precisely. If passed without setting -rotation, the rotation will be assumed to be zero, and -rotationbnds will be the +- bound around 0 degrees. Units in degrees. The smaller the range, the faster the solve speed. If not supplied then the solver automatically estimates the rotation and also its upper and lower estimate bounds; the auto-estimator mitigates the need for the user to provide a rotation estimate, but the user may still do so. '))
    parser.add_argument('-buffer', help=formatting('Tolerance buffer around image field, in arcminutes. This field can be negative, if one wishes to mitigate image padding in the query.'))
    parser.add_argument('-fieldshape', help=formatting('Shape of field to query: "rectangle" (default) or "circle". Circle may only be used if pixwidth and pixheight are equal. Square query uses a polygon query with corners defined by an ad-hoc WCS given the supplied field parameters, whereas circle uses a radius.'))
    parser.add_argument('-catalogue', help=formatting('This specifies the catalogue to query for sources in the given nominal image field. The default is Gaia DR3, in its \'g\' band. To specify other filter bands, see the -filter option setting. Possible catalogues options are: \n\n \'GaiaDR3\'.'))
    parser.add_argument('-filter', help=formatting('This specifies the filter to use from a given catalogue. Default catalogue and filter are Gaia DR3 in \'g\' band. If a specified filter isn\'t found in a given catalogue (specified or default), an error is thrown. Valid filters for given catalogues are as follows: \n\nGaia DR3: \'bp\' (blue), \'g\' (green), \'rp\' (red).'))
    parser.add_argument('-pmepoch', help=formatting('Pass the year.year value of the observation to update the RA and Dec entries of the table with their proper motion adjustments, given the catalogue reference epoch. Only entries in the query which have valid proper motion entries will be saved to ouput.'))
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
    parser.add_argument('--verbosity', help=formatting('Sets the verbosity level of console messages. A value of 1 represents normal verbosity and is the default. Set to 0 for a silent run, or 2 for higher verbosity.'))
    parser.add_argument('--debug', action='store_true', help=formatting('Enters debug mode. Steps through the program, pausing at intervals to print diagnostic and debugging info to the console and display various plots and images. A debug report will be created and saved in the current working directory. --verbosity is automatically set to 2.'))

    input_args = parser.parse_args()

    return input_args

def find_WCS(filename=None, ra=None, dec=None, scale=None, scalebnds=None, rotation=None, rotationbnds=None, buffer=None, shape=None, catalogue=None, pmepoch=None, filter=None, npts=None, nrefinepts=None, pixsat=None, kernelrad=None, sourcesep=None, vertextol=None, nmatchpoints=None, nmatchpercent=None, save=None, load=None, verbosity=None, debug=None):

    user_dir = str(Path.cwd())

    if verbosity is None:
        verbosity = 1
    else:
        try:
            verbosity = int(verbosity)
            if 0 <= verbosity <= 2:
                pass
            else:
                sys.exit('ERROR: verbosity must be between 0 and 2.')            
        except:
            sys.exit('ERROR: verbosity must be an integer.')        

    if debug is True:
        verbosity = 2
        
    if verbosity >= 1:
        print("\nParsing arguments...")

    if load is None:
        if filename is None:
            sys.exit('ERROR: -filename is required.')          
        else:
            try:
                filename = str(filename)
                filepath = user_dir+'\\'+filename
                if verbosity >= 1:
                    print("| Opening fits image...")
                hdul = fits.open('{}'.format(filepath))
                img = hdul[0].data.astype(np.double)
                header = hdul[0].header
                if verbosity >= 1:
                    print("| done")
            except Exception as e:
                sys.exit('ERROR: Could not open the specified FITS file. Check that you are in the directory containing your FITS files.\nORIGINAL ERROR: {}'.format(e))
                
        if ra is None:
            sys.exit('ERROR: -ra is required.')          
        else:
            if len(ra.split(':')) == 1:
                try:
                    ra = float(ra)              ###decimal
                    if 0 <= ra < 360:
                        pass
                    else:
                        sys.exit('ERROR: ra in degree.decimal format must be between 0 and 360.')                      
                except:                            
                    if ra in header:       ###header keywords
                        ra = float(header['{}'.format(ra)])
                    else:
                        sys.exit('ERROR: Could not find the specified keyword in the FITS header (assuming ra was given as a keyword).')                      
            elif len(ra.split(':')) == 3:      ###sexagesimal
                ra_hms = ra.split(':')
                try:
                    hrs = int(ra_hms[0])
                    mins = int(ra_hms[1])
                    secs = float(ra_hms[2])
                    if 0 <= hrs < 24 and  0 <= mins < 60 and 0 <= secs < 60:
                        ra = (hrs + mins/60 + secs/3600)*15
                    else:
                        sys.exit('ERROR: ra in HH:MM:SS.S format must hrs between 0 and 24, minutes between 0 and 60, and seconds between 0 and 60)')                     
                except:
                    sys.exit('ERROR: ra in HH:MM:SS.S format must have integer hours, integer minutes, and integer or decimal seconds.')             
            else:
                sys.exit('ERROR: ra must be entered either in HH:MM:SS.S format (e.g. "18:44:27.891"), degree.decimal format (e.g. "78.132"), or as a FITS header keyword whose value will be retrieved (e.g. "RA_PT").')
                
        if dec is None:
            sys.exit('ERROR: -dec is required.')         
        else:
            if len(dec.split(':')) == 1:
                try:
                    dec = float(dec)                 ###decimal
                    if -90 <= dec < 90:
                        pass
                    else:
                        sys.exit('ERROR: dec in degree.decimal format must be between -90 and 90.')                     
                except:
                    if dec in header:        ###header keywords
                        dec = float(header['{}'.format(dec)])
                    else:
                        sys.exit('ERROR: Could not find the specified keyword in the FITS header (assuming dec was given as a keyword).')                      
            elif len(dec.split(':')) == 3:     ###sexagesimal
                dec_hms = dec.split(':')
                try:
                    degs = int(dec_hms[0])
                    amins = int(dec_hms[1])
                    asecs = float(dec_hms[2])
                    if -90 <= degs < 90 and 0 <= amins < 60 and 0 <= asecs < 60:
                        dec = degs + amins/60 + asecs/3600
                    else:
                        sys.exit('ERROR: dec in DD:MM:SS.S" format must have degrees between -90 and 90, arcminutes between 0 and 60, and arcseconds between 0 and 60.')                      
                except:
                    sys.exit('ERROR: dec in DD:MM:SS.S" format must have integer degrees, integer arcminutes, and integer or decimal arcseconds.')                
            else:
                sys.exit('ERROR: dec must be entered either in DD:MM:SS.S" format (e.g. "-49:12:09.255"), degree.decimal format (e.g. "78.132"), or as a FITS header keyword whose value will be retrieved (e.g. "DEC_PT").')             

        if scale is None:
            sys.exit('ERROR: scale is required.')          
        else:
            try:
                scale = float(scale)
                if scale > 0:
                    pass
                else:
                    sys.exit('ERROR: scale must be greater than 0.')                  
            except:
                sys.exit('ERROR: scale must be a number.')
                
        if scalebnds is None:
            scalebnds = 5.
        else:
            try:
                scalebnds = float(scalebnds)
                if scalebnds >= 0:
                    pass
                else:
                    sys.exit('ERROR: scalebnds must be greater than 0.')                 
            except:
                    sys.exit('ERROR: scalebnds must be a number.')                

        if rotation is None and rotationbnds is None:
            rotation = None       #auto
            rotationbnds = None        #auto
        elif rotation is None and rotationbnds is not None:
            try:
                rotationbnds = float(rotationbnds)
                if 0 <= rotationbnds <= 180:
                    rotation = None        #auto
                else:
                    sys.exit('ERROR: rotationbnds must be between 0 and 180.')                
            except:
                sys.exit('ERROR: rotationbnds must be a number.')             
        elif rotation is not None and rotationbnds is None:
            try:
                rotation = float(rotation)
                if -180 <= rotation < 180:
                    rotationbnds = None       #auto
                else:
                    sys.exit('ERROR: rotation must be between -180 and 180.')                 
            except:
                sys.exit('ERROR: rotation must be a number.')              
        elif rotation is not None and rotationbnds is not None:
            try:
                rotation = float(rotation)
                if -180 <= rotation < 180:
                    pass
                else:
                    sys.exit('ERROR: rotation must be between -180 and 180.')                   
            except:
                sys.exit('ERROR: rotation must be a number.')             
            try:
                rotationbnds = float(rotationbnds)
                if 0 <= rotationbnds <= 180:
                    pass
                else:
                    sys.exit('ERROR: rotationbnds must be between 0 and 180.')                
            except:
                sys.exit('ERROR: rotationbnds must be a number')
                
        if buffer is None:
            buffer = 0
        else:
            try:
                buffer = float(buffer)
            except:
                sys.exit("ERROR: buffer must be a number.")

        if shape is None:
            shape = 'rectangle'
        else:
            if shape in ['rectangle','circle']:
                if shape == 'circle':
                    if header['NAXIS1'] == header['NAXIS2']:
                        pass
                    else:
                        sys.exit("ERROR: circle query requires a square image, i.e., NAXIS1 = NAXIS2.")
            else:
                sys.exit("ERROR: shape must be from the following options: ['rectangle','circle']")

        if catalogue is None:
            catalogue = 'GaiaDR3'
            if filter is None:
                filter = 'g'
            elif filter in ['bp','g','rp']:
                pass
            else:
                sys.exit("ERROR: filter for the GaiaDR3 catalog must be from the following options: ['bp','g','rp'].")
        elif catalogue in ['GaiaDR3']:
            catalogue = catalogue
            if catalogue == 'GaiaDR3':
                if filter is None:
                    filter = 'g'
                elif filter in ['bp','g','rp']:
                    pass
                else:
                    sys.exit("ERROR: filter for the GaiaDR3 catalog must be from the following options: ['bp','g','rp'].")        
        else:
            sys.exit("ERROR: catalogue must be from the following options: ['GaiaDR3'].")

        if pmepoch is None:
            pmepoch = 0
        else:
            try:
                pmepoch = float(pmepoch)
                if pmepoch >= 0:
                    pass
                else:
                    sys.exit("pmepoch")
            except:
                sys.exit("ERROR: pmepoch must be a positive number.")

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
                    sys.exit("ERROR: npts must be greater than or equal to 10.")
            except:
                sys.exit('ERROR: npts must be an integer.')
                
        if nrefinepts is None:
            nrefinepts = 500
        else:
            try:
                nrefinepts = int(nrefinepts)
                if npts <= nrefinepts <= 1000:
                    pass
                else:
                    sys.exit("ERROR: nrefinepts must be between npts and 1000.")     
            except:
                sys.exit('ERROR: nrefinepts must be an integer.')
                
        if pixsat is None:
            pixsat = 0.0
        else:
            try:
                pixsat = float(pixsat)
                if pixsat >= 0: 
                    pass
                else:
                    sys.exit("ERROR: pixsat must be greater than or equal to 0.")      
            except:
                sys.exit('ERROR: pixsat must be a number.')
                
        if kernelrad is None:
            kernelrad = 2
        else:
            try:
                kernelrad = int(kernelrad)
                if kernelrad > 2:
                    pass
                else:
                    sys.exit("ERROR: kernelrad must be greater than or equal to 2.")    
            except:
                sys.exit("ERROR: kernelrad must be an integer.")
                
        if sourcesep is None:
            sourcesep = 25
        else:
            try:
                sourcesep = int(sourcesep)
                if kernelrad <= sourcesep <= 45:
                    pass
                else:
                    sys.exit("ERROR: sourcesep must be greater than or equal to the supplied kernelrad and less than or equal to 45.")  
            except:
                sys.exit("ERROR: sourcesep must be an integer.")
                
        if vertextol is None:
            vertextol = 0.25
        else:
            try:
                vertextol = float(vertextol)
                if 0 < vertextol < 2:
                    pass
                else:
                    sys.exit("ERROR: vertextol must be between 0 and 2.")
            except:
                sys.exit("ERROR: vertextol must be a number.")
                
        if nmatchpoints is None:
            nmatchpoints = 6
        else:
            try:
                nmatchpoints = int(nmatchpoints)
                if nmatchpoints > 2:
                    pass
                else:
                    sys.exit("ERROR: nmatchpoints must be greater than 2.") 
            except:
                sys.exit("ERROR: nmatchpoints must be an integer.")        

        if nmatchpercent is None:
            nmatchpercent = 25
        else:
            try:
                nmatchpercent = float(nmatchpercent)
                if nmatchpercent > 10:
                    pass
                else:
                    sys.exit("ERROR: nmatchpercent must be greater than 10.")
            except:
                sys.exit("ERROR: nmatchpercent must be a number.")
                
        if save is None:
            pass
        else:
            if verbosity >= 1:
                print("Saving args to file...")
            try:
                contents = {
                    'filename' : filename,
                    'ra' : ra,
                    'dec' : dec,
                    'scale' : scale,
                    'scalebnds' : scalebnds,
                    'rotation' : rotation,
                    'rotationbnds' : rotationbnds,
                    'buffer' : buffer,
                    'fieldshape' : shape,
                    'catalogue' : catalogue,
                    'filter' : filter,
                    'pmepoch' : pmepoch,
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
                    if verbosity >= 1:
                        print("Creating {}\\cmd_args".format(user_dir))
                    Path('{}\\cmd_args'.format(user_dir)).mkdir(parents=True)
                with open('{}\\cmd_args\\saveoptions_{}.json'.format(user_dir,id),'w') as f:
                    f.write(jobject)
                if verbosity >= 1:
                    print("done. Saved as saveoptions_{}.json.".format(id))
            except Exception as e:
                print(e)
                sys.exit("ERROR: Could not save options.")          
    else:
        try:
            if verbosity >= 1:
                print("Loading arguments from saveoptions_{}.json...".format(id))
            with open('{}\\cmd_args\\saveoptions_{}.json'.format(user_dir,id),'r') as f:
                jobject = json.load(f)
                filename = jobject['filename']
                ra = jobject['ra']
                dec = jobject['dec']
                scale = jobject['scale']
                scalebnds = jobject['scalebnds']
                rotation = jobject['rotation']
                rotationbnds = jobject['rotationbnds']
                buffer = jobject['buffer']
                shape = jobject['fieldshape']
                catalogue = jobject['catalogue']
                filter = jobject['filter']
                pmepoch = jobject['pmepoch']
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
                if verbosity >= 1:
                    print("done")
        except Exception as e:
            print(e)
            sys.exit("ERROR: Could not load options. Check that a json file exists with the specified id.")    

    if verbosity >= 1:
        print("done")

    filename_body = ''
    namepieces = filename.split('.')
    for n in range(len(namepieces)-1):
        filename_body += namepieces[n]
    filename_ext = namepieces[-1]

    debug_report = ''
    if debug:
        if not Path('{}\\debug'.format(user_dir)).is_dir():
            if verbosity >= 1:
                print("Creating {}\\debug".format(user_dir))
            Path('{}\\debug'.format(user_dir)).mkdir(parents=True)  
        debug_report = filename_body+datetime.now().strftime("_%y-%m-%d_%H-%M-%S")
        Path('{}\\debug\\{}'.format(user_dir,debug_report)).mkdir(parents=True)

    #################### GLOSSARY OF IMPORTANT VARIABLE NAMES ####################
    ###
    ###     img:                    The FITS image supplied by the user. The PSE function extracts the point sources in the image and puts their metadata (namely, coordinates and brightness) into pse_metadata ("source metadata").
    ###     srcindexmap:            The source index map. This is a copy of the image with every point set to a value of -1 initially. In the PSE function, circular areas (the kernel) around each found source are given
    ###                             non-zero values. This provides a way of checking if an inverse-transformed point in the WCS function "lands" on a source.
    ###     pse_metadata:           The source metadata array, which stores the coordinates, brightness, and background of every found source from the PSE (up to nrefinepts).           
    ###     allintrmcoords_view:    An array holding the projected coordinates ("intermediate coordinates") of a set of sky coordinates downloaded from the catalogue. The success of the WCS is contingent on the geometric
    ###                             correspondence between the intermediate coordinates and the pse coordinates.
    ###     psecoords_view:         An array holding the pixel coordinates of the npts brightest sources from the pse_metadata. The "view" suffix indicates that it is a "memory view", a type of high-efficency array
    ###                             pointer datatype in Cython.
    ###     intrmcoords_view:       The equivalent of the above except for holding intermediate coordinates.                 
    ###     S                       The scale parameter found by the optimizer when finding the initial solution in the WCS.
    ###     phi                     The rotation parameter found by the optimizer when finding the initial solution in the WCS.
    ###     CRPIXx                  The reference pixel location's x coordinate, also found by the optimizer in the WCS. (Once for the initial solution, and then improved
    ###                             in the refinement stage). CRPIX is the point in the image distinguished by the property that vectors drawn from it to the psecoords only need
    ###                             to undergo a rotation and a scaling (no translation) in order to land on the corresponding intermediate coords.
    ###     CRPIXy                  The reference pixel location's y coordinate.
    ###     CD1_1                   CD matrix element that equals S*cos(phi). All elements are found when refining the initial solution in the WCS. The complete CD matrix, together with CRPIX, is the final solution.
    ###     CD1_2                   CD matrix element that equals -S*sin(phi).
    ###     CD2_1                   CD matrix element that equals S*sin(phi).
    ###     CD2_2                   CD matrix element that equals S*cos(phi).
    #####################################################################

    if rotation is not None:
        rotation = rotation*pi/180
    if rotationbnds is not None:
        rotationbnds = rotationbnds*pi/180
    img_xmax = int(img.shape[1])
    img_ymax = int(img.shape[0])
    radius = scale/3600*img_xmax/2 + buffer/60 #degrees
    pixelradius = radius/(scale/3600) #pixels

    srcindexmap_initial = -1*np.ones(img.shape,dtype=np.int32)
    srcindexmap_refine = -1*np.ones(img.shape,dtype=np.int32)
    pse_metadata = np.zeros((nrefinepts,3),dtype=np.double)
    pse_metadata_inv = np.zeros((nrefinepts,3),dtype=np.double)
    
    if verbosity >= 1:
        print("\nGetting sources in image...")
    num_psesources = PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, pixelradius, shape, srcindexmap_initial, srcindexmap_refine, pse_metadata, user_dir, debug_report, filename, verbosity, debug)
    if verbosity >= 1:
        print("done")

    pse_metadata_inv[:,0] = img_xmax - pse_metadata[:,0]
    pse_metadata_inv[:,1] = img_ymax - pse_metadata[:,1]
    pse_metadata_inv[:,2] = pse_metadata[:,2]

    allintrmcoords = np.zeros((nrefinepts,2))
    meancatcoords = np.zeros(2)

    if verbosity >= 1:
        print("\nGetting sources from catalogue...")
    num_catsources = getIntermediateCoords(ra, dec, scale, img_xmax, img_ymax, shape, filter, catalogue, pmepoch, nrefinepts, allintrmcoords, meancatcoords, user_dir, debug_report, verbosity, debug)
    if verbosity >= 1:
        print("done")

    scale = scale/3600*pi/180
    scalebnds = .01*scalebnds
    vertextol = vertextol*pi/180

    if nmatchpoints < nmatchpercent*npts:
        minmatches = nmatchpoints
    else:
        minmatches = nmatchpercent*npts
    kerneldiam = kernelrad*2+1

    headervals = np.zeros(12)

    if verbosity >= 1:
        print("\nPerforming WCS...")
    WCS(scale, scalebnds, rotation, rotationbnds, npts, nrefinepts, vertextol, allintrmcoords, meancatcoords, pse_metadata, pse_metadata_inv, srcindexmap_initial, srcindexmap_refine, img_xmax, img_ymax, minmatches, kerneldiam, num_psesources, num_catsources, headervals, user_dir, debug_report, filename, verbosity, debug)
    if verbosity >= 1:
        print("done")

    header['CTYPE1'] = ('RA---TAN', 'WCS type of horizontal coord. transformation')
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
    
    outfilename = filename_body + ' WCS.' + filename_ext
    numbered_outfilename = insertCopyNumber(outfilename,filename)
    outfilepath = user_dir+'\\'+numbered_outfilename
    if verbosity >= 1:
        print("\nWriting to {}...".format(numbered_outfilename))
    fits.writeto('{}'.format(outfilepath), img, header, overwrite=True, output_verify='silentfix')
    if verbosity >= 1:
        print("done")

    print("\nfastrometry completed successfully.")

    return

def fastrometryCall():
    args = getArgsFromCommandLine()
    solution = find_WCS(filename=args.filename, ra=args.ra, dec=args.dec, scale=args.scale, scalebnds=args.scalebnds, rotation=args.rotation, rotationbnds=args.rotationbnds, buffer=args.buffer, shape=args.fieldshape, catalogue=args.catalogue, filter=args.filter, pmepoch=args.pmepoch, npts=args.npts, nrefinepts=args.nrefinepts, pixsat=args.pixsat, kernelrad=args.kernelrad, sourcesep=args.sourcesep, vertextol=args.vertextol, nmatchpoints=args.nmatchpoints, nmatchpercent=args.nmatchpercent, save=args.save, load=args.load, verbosity=args.verbosity, debug=args.debug)

    