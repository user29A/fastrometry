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
from textwrap import wrap
import json
from math import pi
import sys

from astropy.io.fits import writeto
from numpy import savetxt

################
VERSION="1.0.5"
################

def insertCopyNumber(outfilename, diagnostic, filename_body, filename_ext):
    copynum = 1
    while Path(outfilename).is_file():
        outfilename = filename_body+' WCS ({})'.format(copynum)+filename_ext
        diagnostic = filename_body+' WCS ({}) diagnostic.csv'.format(copynum)
        copynum += 1
    return outfilename, diagnostic

def formatting(text,width,subseq_indents): #Thank you to user blackpen on StackExchange for sharing this function
    text=re.sub('\s+',' ',text); text=re.sub('^\s+','',text); text=re.sub('\s+$','',text)
    text=wrap(text,width=width,initial_indent=' '*4,subsequent_indent=' '*subseq_indents)
    s=""
    for i in (text): s=s+i+"\n"
    s=re.sub('\s+$','',s)
    return(s+'\n \n')

def getCMDargs():
    indent_formatter = lambda prog: argparse.RawTextHelpFormatter(prog,max_help_position=40)
    parser = argparse.ArgumentParser(description='Astrometry package written in Python.\n\nRequired parameters are:\n\n> filename\n> -scale\n> -ra\n> -dec',usage=formatting("fastrometry filename -ra -dec -scale [-scalebnds] [-rotation] [-rotationbnds] [-buffer] [-fieldshape] [-catalogue] [-filter] [-pmepoch] [-npts] [-nrefinepts] [-pixsat] [-kernelrad] [-sourcesep] [-vertexol] [-nmatchpoints] [-nmatchpercent] [-wcsdiagnostics] [--save] [--load] [--version] [--verbosity] [--debug] [--help]",100,10),add_help=False,formatter_class=indent_formatter)
    parser.add_argument('filename', type=Path, help=formatting('The FITS image file to solve the world coordinate solution for.',80,4))
    parser.add_argument('-ra', required=True, help=formatting('The approximate right-ascension of the field center of the image. This can be supplied in either right-ascension sexagesimal (HH:MM:SS.S) format, degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either right-ascension sexagesimal or degree.decimal format.',80,4))
    parser.add_argument('-dec', required=True, help=formatting('The approximate declination of the field center of the image. This can be supplied in either declination sexagesimal format (DD:MM:SS.S), degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either declination sexagesimal or degree.decimal format.',80,4))
    parser.add_argument('-scale', required=True, help=formatting('This is the approximate field scale, in arcseconds per pixel.',80,4))
    parser.add_argument('-scalebnds', help=formatting('This is the "plus or minus" range of the field scale, in the same units as the field scale of arcseconds per pixel. If no scalebnds are supplied then a +-5%% range bound is assumed. Zero is a valid option if the scale is known precisely, and overrides the +-5%% assumption default, and will increase solve speed.',80,4))
    parser.add_argument('-rotation', help=formatting('Use this to provide an initial estimate of the image field rotation relative to sky coordinates, between +- 180 degrees. Units in degrees. Zero degrees corresponds to the CAST convention, as does positive angle measure, and negative angle is opposite rotation to that. If not supplied then the solver automatically estimates the rotation and also its upper and lower estimate bounds; the auto-estimator mitigates the need for the user to provide a rotation estimate, but the user may still do so.',80,4))
    parser.add_argument('-rotationbnds', help=formatting('This is the "plus or minus" range bounds of the rotation estimate. Zero is a legitimate value if the rotation is known precisely. If passed without setting -rotation, the rotation will be assumed to be zero, and -rotationbnds will be the +- bound around 0 degrees. Units in degrees. The smaller the range, the faster the solve speed. If not supplied then the solver automatically estimates the rotation and also its upper and lower estimate bounds; the auto-estimator mitigates the need for the user to provide a rotation estimate, but the user may still do so.',80,4))
    parser.add_argument('-buffer', help=formatting('Tolerance buffer around image field, in arcminutes. This field can be negative, if one wishes to mitigate image padding in the query.',80,4))
    parser.add_argument('-fieldshape', help=formatting('Shape of field to query: "rectangle" (default) or "circle". Circle may only be used if pixwidth and pixheight are equal. Square query uses a polygon query with corners defined by an ad-hoc WCS given the supplied field parameters, whereas circle uses a radius.',80,4))
    parser.add_argument('-catalogue', help=formatting('This specifies the catalogue to query for sources in the given nominal image field. The default is Gaia DR3, in its \'g\' band. To specify other filter bands, see the -filter option setting. Possible catalogues options are: \n\n \'GaiaDR3\'.',80,4))
    parser.add_argument('-filter', help=formatting('This specifies the filter to use from a given catalogue. Default catalogue and filter are Gaia DR3 in \'g\' band. If a specified filter isn\'t found in a given catalogue (specified or default), an error is thrown. Valid filters for given catalogues are as follows: \n\nGaia DR3: \'bp\' (blue), \'g\' (green), \'rp\' (red).',80,4))
    parser.add_argument('-pmepoch', help=formatting('Pass the year.year value of the observation to update the RA and Dec entries of the table with their proper motion adjustments, given the catalogue reference epoch. Only entries in the query which have valid proper motion entries will be saved to ouput.',80,4))
    parser.add_argument('-npts', help=formatting('The number of sources to use in both the catalogue and image source extraction. Suggest 25 for good wavelength correspondence between the catalogue and image; 50 for poor correspondence; 100 for very poor; 200 for a hope and a prayer and you might crash your system...but it still works often. 75 is the default when not supplied.',80,4))
    parser.add_argument('-nrefinepts', help=formatting('This is the number of points to use for automatic refinement of the initial solution based on npts; npts must be a relatively smaller value (see comment on npts) given the (n-choose-3)^2 nature of the solver, but the initial solution may be easily refined with a much larger number of points. Default is 500; maximum is 1000; minimum is npts.',80,4))
    parser.add_argument('-pixsat', help=formatting('This is important and very helpful to use if the image is known to have saturated pixels. One can inform the point source extractor the threshold above which sources will be expected to be saturated and form island-plateaus instead of Gaussian peaks. For example, a pixel-well may saturate at an ADU value of ~50,000, although it might have a 16-bit ADC which would saturate at 65,536 pixel value. In this case a \'pixsat\' should be specified as, say, 45,000, just below the ADU well-depth. If -pixsat is not supplied, there will be no special treatment of saturated sources.',80,4))
    parser.add_argument('-kernelrad', help=formatting('This is the radius of the centroiding kernel used in the point-source extractor. Default, and minimum, is 2, which means that +-2 pixels about a source peak will be used for centroiding the source, i.e., a 5x5 pixel array centered on the source peak is centroided. If the sources are over-sampled then the user might like to set a larger radius such that the kernel samples the complete PSF. The default of 2 assumes that the image source PSFs are critically-sampled.',80,4))
    parser.add_argument('-sourcesep', help=formatting('This sets the minimum separation allowed between sources the image, in unit pixels. Default is 25. If it is less than the centroid kernel radius, it is made equal to the centroid kernel radius.',80,4))
    parser.add_argument('-vertextol', help=formatting('Default 0.25 degrees. This sets the vertex tolerance within which two vertices of a triangle must match in angle to another triangle in order to be considered a potential pattern-match between the image source coordinates and catalogue source coordinates. Not much testing has been done on this option and 0.25 has always been used. A smaller tolerance will be less forgiving to distortion but would offer faster searching and fewer false-positives, whereas a larger tolerance will be more forgiving to distortion at the expense of slower searching and a higher rate of false-positives.',80,4))
    parser.add_argument('-nmatchpoints', help=formatting('This specifies the number of points which must match between the point source list and catalogue list to be considered a solution. Default is 6 points. Minimum is 3 but this is very likely to produce false-positive solutions. 4 or 5 can be used where there are simply very few points in the source image. 6 or higher should guarantee no false positives.',80,4))
    parser.add_argument('-nmatchpercent', help=formatting('Same as above but as a percentage of point sources available. Default is 25%%. If both nmatchpoints and nmatchpercent are supplied, the smaller threshold will be used, found by comparing nmatchpoints to nmatchpercent*npts.',80,4))
    parser.add_argument('-wcsdiagnostics', action='store_true', help=formatting('If this option is provided, fastrometry will write an additional csv file beside the FITS image file which provides diagnostic information about the WCS solution. The PSE centroids, their sky coordinate values via the WCS solution, the corresponding sky coordinates from the catalogue, and then the differentials, are written to the csv diagnostic file.',80,4))
    parser.add_argument('--save', help=formatting('Save the options settings as a file to disk to recall later. User supplies an ID, for example: "--save UVIT". Useful if frequently processing different telescopic fields, allowing quick input of WCS solver settings. Saves all settings EXCEPT for filename, ra, dec, pmepoch, so that different images but taken from the same telescope can be processed at their unique individual characteristics.',80,4))
    parser.add_argument('--load', help=formatting('Load an options settings configuration. User supplies an ID that matches that of a previously-saved file. Useful when frequently processing the same telescopic field.',80,4))
    parser.add_argument('--version', '-V', action='version', version='fastrometry {}'.format(VERSION), help=formatting('Shows the current version of fastrometry.',80,4))
    parser.add_argument('--verbosity', help=formatting('Sets the verbosity level of print messages. A value of 1 represents normal verbosity and is the default. Set to 0 for a silent run, or 2 for higher verbosity.',80,4))
    parser.add_argument('--debug', action='store_true', help=formatting('Enters debug mode. Steps through the program, pausing at intervals to print diagnostic and debugging info to the print and display various plots and images. A debug report will be created and saved in the current working directory. --verbosity is automatically set to 2.',80,4))
    parser.add_argument('--h', '--help', '-h', '-help', action='help', default=argparse.SUPPRESS, help=formatting('Shows this help message.',80,4))

    input_args = parser.parse_args()

    return input_args

def validateVerbosityAndDebug(verbosity, debug):
    if verbosity is None:
        verbosity = 1
    else:
        try:
            verbosity = int(verbosity)
        except:
            sys.exit("ERROR: verbosity must be an integer.")
        if not 0 <= verbosity <= 2:
            sys.exit("ERROR: verbosity must be between 0 and 2.")
    
    if debug is None:
        debug = False
    else:
        try:
            debug = bool(debug)
        except:
            sys.exit("ERROR: --debug must be a boolean.")
        if debug is True:
            verbosity = 2

    return verbosity, debug

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

writeto = printEvent(writeto)
savetxt = printEvent(savetxt)

def printItem(message, item, verbosity, levelneeded):
    if verbosity >= levelneeded:
        print('--> '+message+' {}'.format(item))

def printMessage(message, verbosity, levelneeded):
    if verbosity >= levelneeded:
        print(message)

@printEvent
def validateFilename(filename):
    filename = Path(filename)
    filepath = filename.resolve()
    if not filepath.is_file():
        sys.exit("ERROR: could not open the specified FITS file. Check that you are in the directory containing your FITS files.")
    user_dir = filepath.parents[0]
    filename_body = filename.stem
    filename_ext = filename.suffix

    return filepath, user_dir, filename_body, filename_ext

@printEvent
def openFITS(filepath):
    with fits.open(filepath) as hdul:
        img = hdul[0].data.astype(np.double)
        header = hdul[0].header
    return img, header


@printEvent
def validateOptions(header, load, options):

    ##Since argparse outputs all arguments as strings, we have to test type by 'trying' type conversions

    ra = options[0]
    try:    ## if ra is a float: ...
        ra = float(ra)
        if not 0 <= ra <= 360:  
            sys.exit("ERROR: -ra must be between 0 and 360 (assuming -ra was given in degree.decimal format).")
    except: ## else: ...
        parts = len(ra.split(':'))
        if parts == 1:
            if ra not in header:
                sys.exit("ERROR: could not find the specified keyword in the FITS header (assuming -ra was supplied as a keyword).")
            ra = float(header[ra])
        elif parts == 3:
            hrs = parts[0]
            mins = parts[1]
            secs = parts[2]
            try:
                hrs = int(hrs)
            except:
                sys.exit("ERROR: hours must be an integer (assuming -ra was given in sexigesimal format HH:MM:SS.S).")
            try:
                mins = int(mins)
            except:
                sys.exit("ERROR: minutes must be an integer (assuming -ra was given in sexigesimal format HH:MM:SS.S).")
            try:
                secs = float(secs)
            except:
                sys.exit("ERROR: seconds must be a number (assuming -ra was given in sexigesimal format HH:MM:SS.S).")
            if not 0 <= hrs <= 24:
                sys.exit("ERROR: hours must be between 0 and 24 (assuming -ra was given in sexigesimal format HH:MM:SS.S).")
            if not 0 <= mins <= 60:
                sys.exit("ERROR: minutes must be between 0 and 60 (assuming -ra was given in sexigesimal format HH:MM:SS.S).")
            if not 0 <= secs <= 60: 
                sys.exit("ERROR: seconds must be between 0 and 60 (assuming -ra was given in sexigesimal format HH:MM:SS.S).")
    options[0] = ra

    dec = options[1]
    try:
        dec = float(dec)
        if not 0 <= dec <= 360:  
            sys.exit("ERROR: -dec must be between 0 and 360 (assuming -dec was given in degree.decimal format).")
    except:
        parts = len(dec.split(':'))
        if parts == 1:
            if dec not in header:
                sys.exit("ERROR: could not find the specified keyword in the FITS header (assuming -dec was supplied as a keyword).")
            dec = float(header[dec])
        elif parts == 3:
            hrs = parts[0]
            mins = parts[1]
            secs = parts[2]
            try:
                hrs = int(hrs)
            except:
                sys.exit("ERROR: hours must be an integer (assuming -dec was given in sexigesimal format HH:MM:SS.S).")
            try:
                mins = int(mins)
            except:
                sys.exit("ERROR: minutes must be an integer (assuming -dec was given in sexigesimal format HH:MM:SS.S).")
            try:
                secs = float(secs)
            except:
                sys.exit("ERROR: seconds must be a number (assuming -dec was given in sexigesimal format HH:MM:SS.S).")
            if not 0 <= hrs <= 24:
                sys.exit("ERROR: hours must be between 0 and 24 (assuming -dec was given in sexigesimal format HH:MM:SS.S).")
            if not 0 <= mins <= 60:
                sys.exit("ERROR: minutes must be between 0 and 60 (assuming -dec was given in sexigesimal format HH:MM:SS.S).")
            if not 0 <= secs <= 60: 
                sys.exit("ERROR: seconds must be between 0 and 60 (assuming -dec was given in sexigesimal format HH:MM:SS.S).")
    options[1] = dec

    pmepoch = options[10]
    if pmepoch is None:
        pmepoch = 0
    else:
        try:
            pmepoch = float(pmepoch)
        except:
            sys.exit("ERROR: -pmepoch must be a number.")
        if not 1950 <= pmepoch:
            sys.exit("ERROR: -pmepoch must be greater than or equal to 1950.")
    options[10] = pmepoch

    ##If we are later loading from json, no need to validate the remaining options,
    ##which are just going to be replaced.

    if load is not None:
        return

    scale = options[2]
    if scale is None:
        sys.exit('ERROR: -scale is required.')          
    try:
        scale = float(scale)        
    except:
        sys.exit('ERROR: -scale must be a number.')
    if not scale > 0:
        sys.exit('ERROR: -scale must be greater than 0.')
    options[2] = scale
                
    scalebnds = options[3]
    if scalebnds is None:
        scalebnds = 5
    else:
        try:
            scalebnds = float(scalebnds)                 
        except:
            sys.exit('ERROR: -scalebnds must be a number.')
        if not scalebnds >= 0:
            sys.exit('ERROR: -scalebnds must be greater than 0.')
    options[3] = scalebnds              

    rotation = options[4]
    rotationbnds = options[5]
    if rotation is None and rotationbnds is None:
        pass
    elif rotation is None and rotationbnds is not None:
        try:
            rotationbnds = float(rotationbnds)           
        except:
            sys.exit('ERROR: -rotationbnds must be a number.')             
        if not 0 <= rotationbnds <= 180:
            sys.exit('ERROR: -rotationbnds must be between 0 and 180.')            
    elif rotation is not None and rotationbnds is None:
        try:
            rotation = float(rotation)             
        except:
            sys.exit('ERROR: -rotation must be a number.')
        if not -180 <= rotation < 180:
            sys.exit('ERROR: -rotation must be between -180 and 180.')  
    elif rotation is not None and rotationbnds is not None:
        try:
            rotation = float(rotation)  
        except:
            sys.exit('ERROR: -rotation must be a number.')      
        if not -180 <= rotation < 180:
            sys.exit('ERROR: -rotation must be between -180 and 180.')             
        try:
            rotationbnds = float(rotationbnds)
        except:
            sys.exit('ERROR: -rotationbnds must be a number')
        if not 0 <= rotationbnds <= 180:
            sys.exit('ERROR: -rotationbnds must be between 0 and 180.')
    options[4] = rotation
    options[5] = rotationbnds
    
    buffer = options[6]
    if buffer is None:
        buffer = 0
    else:
        try:
            buffer = float(buffer)
        except:
            sys.exit("ERROR: -buffer must be a number.")
    options[6] = buffer

    shape = options[7]
    if shape is None:
        shape = 'rectangle'
    else:
        if shape not in ['rectangle','circle']:
            sys.exit("ERROR: -shape must be from the following options: ['rectangle','circle']")
        if shape == 'circle':
            if not header['NAXIS1'] == header['NAXIS2']:
                sys.exit("ERROR: circle query requires a square image, i.e., NAXIS1 = NAXIS2.")
    options[7] = shape

    catalogue = options[8]
    filter = options[9]
    if catalogue is None:
        catalogue = 'GaiaDR3'
    if catalogue == 'GaiaDR3':
        if filter is None:
            filter = 'g'
        else:
            if filter not in ['bp','g','rp']:
                sys.exit("ERROR: -filter for the GaiaDR3 catalogue must be from the following options: ['bp','g','rp'].")
    elif catalogue not in ['GaiaDR3']:
        sys.exit("ERROR: -catalogue must be from the following options: ['GaiaDR3'].")
    options[8] = catalogue
    options[9] = filter

    npts = options[11]
    if npts is None:
        npts = 75
    else:
        try:
            npts = int(npts)
        except:
            sys.exit('ERROR: -npts must be an integer.')
        if not 10 <= npts <= 300:
            sys.exit("ERROR: -npts must be between 10 and 300.")
    options[11] = npts

    nrefinepts = options[12]
    if nrefinepts is None:
        nrefinepts = 500
    else:
        try:
            nrefinepts = int(nrefinepts)
        except:
            sys.exit('ERROR: -nrefinepts must be an integer.')
        if not npts <= nrefinepts <= 1000:
            sys.exit("ERROR: -nrefinepts must be between npts and 1000.")     
    options[12] = nrefinepts
    
    pixsat = options[13]
    if pixsat is None:
        pixsat = 0.
    else:
        try:
            pixsat = float(pixsat)
        except:
            sys.exit('ERROR: -pixsat must be a number.')
        if not pixsat >= 0: 
            sys.exit("ERROR: -pixsat must be greater than or equal to 0.")
    options[13] = pixsat

    kernelrad = options[14]
    if kernelrad is None:
        kernelrad = 2
    else:
        try:
            kernelrad = int(kernelrad)
        except:
            sys.exit("ERROR: -kernelrad must be an integer.")
        if not kernelrad >= 2:
            sys.exit("ERROR: -kernelrad must be greater than or equal to 2.")    
    options[14] = kernelrad
            
    sourcesep = options[15]
    if sourcesep is None:
        sourcesep = 25
    else:
        try:
            sourcesep = int(sourcesep)
        except:
            sys.exit("ERROR: -sourcesep must be an integer.")
        if not kernelrad <= sourcesep <= 45:
            sys.exit("ERROR: -sourcesep must be greater than or equal to -kernelrad and less than or equal to 45.")  
    options[15] = sourcesep

    vertextol = options[16]        
    if vertextol is None:
        vertextol = 0.25
    else:
        try:
            vertextol = float(vertextol)
        except:
            sys.exit("ERROR: -vertextol must be a number.")
        if not 0 < vertextol < 2:
            sys.exit("ERROR: -vertextol must be between 0 and 2.")
    options[16] = vertextol
            
    nmatchpoints = options[17]
    if nmatchpoints is None:
        nmatchpoints = 6
    else:
        try:
            nmatchpoints = int(nmatchpoints)
        except:
            sys.exit("ERROR: -nmatchpoints must be an integer.")
        if not nmatchpoints > 2:
            sys.exit("ERROR: -nmatchpoints must be greater than 2.")
    options[17] = nmatchpoints

    nmatchpercent = options[18]
    if nmatchpercent is None:
        nmatchpercent = 25
    else:
        try:
            nmatchpercent = float(nmatchpercent)
        except:
            sys.exit("ERROR: -nmatchpercent must be a number.")
        if not nmatchpercent > 10:
            sys.exit("ERROR: -nmatchpercent must be greater than 10.")
    options[18] = nmatchpercent

    wcsdiagnostics = options[19]
    if wcsdiagnostics is None:
        wcsdiagnostics = False
    else:
        try:
            wcsdiagnostics = bool(wcsdiagnostics)
        except:
            sys.exit("ERROR: --wcsdiagnostics must be a boolean.")
    options[19] = wcsdiagnostics

@printEvent
def polishID(id):
    re.sub(r'[^\w\-_\. ]', '', id)

@printEvent
def createFolder(user_dir,child):
    (user_dir/child).mkdir(parents=True)

@printEvent
def saveOptions(id, user_dir, options, verbosity):
    try:
        polishID(id, printconsole=("| Checking save ID...","| done",verbosity,1))
        
        contents = {
            'scale' : options[2],
            'scalebnds' : options[3],
            'rotation' : options[4],
            'rotationbnds' : options[5],
            'buffer' : options[6],
            'fieldshape' : options[7],
            'catalogue' : options[8],
            'filter' : options[9],
            'npts' : options[11],
            'nrefinepts' : options[12],
            'pixsat' : options[13],
            'kernelrad' : options[14],
            'sourcesep' : options[15],
            'vertextol' : options[16],
            'nmatchpoints' : options[17],
            'nmatchpercent' : options[18],
            'wcsdiagnostics' : options[19],
        }
        jobject = json.dumps(contents)

        if not (user_dir/'cmd_args').is_dir():
            createFolder(user_dir,"cmd_args",printconsole=("| Creating cmd_args folder...","| done",verbosity,1))

        with open(user_dir/'cmd_args'/'options_{}.json'.format(id),'w') as fp:
            fp.write(jobject)

    except Exception as e:
        print(e)
        sys.exit("ERROR: Could not save options.")          

@printEvent
def loadOptions(id, user_dir, options):
    try:
        with open(user_dir/'cmd_args'/'options_{}.json'.format(id),'r') as fp:
            jobject = json.load(fp)
            options[2] = jobject['scale']
            options[3] = jobject['scalebnds']
            options[4] = jobject['rotation']
            options[5] = jobject['rotationbnds']
            options[6] = jobject['buffer']
            options[7] = jobject['fieldshape']
            options[8] = jobject['catalogue']
            options[9] = jobject['filter']
            options[11] = jobject['npts']
            options[12] = jobject['nrefinepts']
            options[13] = jobject['pixsat']
            options[14] = jobject['kernelrad']
            options[15] = jobject['sourcesep']
            options[16] = jobject['vertextol']
            options[17] = jobject['nmatchpoints']
            options[18] = jobject['nmatchpercent']
            options[19] = jobject['wcsdiagnostics']
    except Exception as e:
        print(e)
        sys.exit("ERROR: Could not load options. Check that a json file exists with the specified id.")    

@printEvent
def convertOptions(options):
    scale = options[2]
    scalebnds = options[3]
    scale = scale/3600*pi/180   #from arcsec/pixel to radians/pixel
    scalebnds = scalebnds*.01   #from % to a decimal
    options[2] = scale
    options[3] = scalebnds

    rotation = options[4]
    rotationbnds = options[5]
    if rotation is not None:
        rotation = rotation*pi/180  #from degrees to radians
    if rotationbnds is not None:
        rotationbnds = rotationbnds*pi/180  #from degrees to radians
    options[4] = rotation
    options[5] = rotationbnds

    buffer = options[6]
    buffer = buffer/60  #from arcminutes to degrees
    options[6] = buffer
    
    vertextol = options[16]
    vertextol = vertextol*pi/180    #from degrees to radians
    options[16] = vertextol

    nmatchpercent = options[18]
    nmatchpercent = nmatchpercent*.01   #from % to a decimal
    options[18] = nmatchpercent

@printEvent
def setupFolders(debug, user_dir, filename_body, verbosity):
    if debug is True:
        debug_folder = (user_dir/'debug')
        if not debug_folder.is_dir():
            createFolder(user_dir, "debug", printconsole=("| Creating debug folder...","| done",verbosity,1))
        report_name = str(filename_body)+str(datetime.now().strftime("_%y-%m-%d_%H-%M-%S"))
        debug_report = (debug_folder/report_name)
        createFolder(debug_folder, report_name, printconsole=("| Creating debug report folder...","| done",verbosity,1))
    else:
        debug_report = None
    gaiaqueries = (user_dir/'gaiaqueries')
    if not gaiaqueries.is_dir():
        createFolder(user_dir, "gaiaqueries", printconsole=("| Creating gaiaqueries folder...","| done",verbosity,1))
    return debug_report, gaiaqueries


def findWCS(filename, ra=None, dec=None, scale=None, scalebnds=None, rotation=None, rotationbnds=None, buffer=None, shape=None, catalogue=None, pmepoch=None, filter=None, npts=None, nrefinepts=None, pixsat=None, kernelrad=None, sourcesep=None, vertextol=None, nmatchpoints=None, nmatchpercent=None, wcsdiagnostics=None, save=None, load=None, verbosity=None, debug=None):

    #Section 1
    print("\n")

    #Validate options, in order needed. Verbosity and debug first, then filename (so we can then open 
    #the FITS file to get the header, needed for validation of other options), then the rest

    verbosity, debug = validateVerbosityAndDebug(verbosity, debug)
    
    filepath, user_dir, filename_body, filename_ext = validateFilename(filename, printconsole=("Validating filename...","done",verbosity,1))
    
    img, header = openFITS(filepath, printconsole=("Opening FITS image...","done",verbosity,1))

    options = [ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, shape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics]   #save and load have no form restrictions and thus do not need to be validated
    printItem("Inputted options:", options, verbosity, 2)

    validateOptions(header, load, options, printconsole=("Validating options...","done",verbosity,1))
    printItem("Options after validating and defaulting:", options, verbosity, 2)

    #Load the "saveable/loadable" the options from json. This only happens if --load was specified. If this
    #is the case, validateOptions will have skipped validating these options to save time. This is also where
    #options can be saved if --save was specified (--save will not have affected which options are validated
    #because all options need to be validated before saving).

    if save is None and load is None:
        pass
    elif save is None and load is not None:
        id = load
        loadOptions(id, user_dir, options, printconsole=("Loading options from saveoptions_{}.json...".format(id),"done",verbosity,1))
        printItem("Options after loading:", options, verbosity, 2)
    elif save is not None and load is None:
        id = save
        saveOptions(id, user_dir, options, verbosity, printconsole=("Saving options to {}.json...".format(id),"done",verbosity,1))
    elif save is not None and load is not None:
        sys.exit("ERROR: cannot specify --save and --load at the same time.")

    #Convert options to correct units for internal operations (e.g., radians instead of degrees).
    convertOptions(options, printconsole=("Converting options...","done",verbosity,1))
    printItem("Options after unit conversions:", options, verbosity, 2)

    #Open Pandora's box
    ra = options[0]
    dec = options[1]
    scale = options[2]
    scalebnds = options[3]
    rotation = options[4]
    rotationbnds = options[5]
    buffer = options[6]
    shape = options[7]
    catalogue = options[8]
    filter = options[9]
    pmepoch = options[10]
    npts = options[11]
    nrefinepts = options[12]
    pixsat = options[13]
    kernelrad = options[14]
    sourcesep = options[15]
    vertextol = options[16]
    nmatchpoints = options[17]
    nmatchpercent = options[18]
    wcsdiagnostics = options[19]

    #Setup the debug folder tree if debug is True. The debug report is a subfolder specific to the
    #active session, and the debug folder contains all the debug_reports.

    debug_report, gaiaqueries = setupFolders(debug, user_dir, filename_body, verbosity, printconsole=("Preparing folders...","done",verbosity,1))

    ############################# GLOSSARY OF IMPORTANT VARIABLE NAMES #############################
    ###
    ###     img:                            A numpy array holding the FITS image supplied by the user.
    ###
    ###     srcindexmap (initial/refine):   The source index map. This has the same dimensions as the image. Every "pixel" (array element) is initially set to -1, corresponding to unoccupied. Pixels
    ###                                     are assigned integer values if they correspond to a source in the image (with smaller integers describing brighter sources). Thus the source index map is 
    ###                                     basically a vacancy map, with integer values assigned to each source kernel, and -1 everywhere else. There are two source index maps: the initial source
    ###                                     index map, and the refine source index map. The difference is that the first holds the npts brightest source kernels, and the second holds all nrefinepts
    ###                                     source kernels.
    ###                                     By checking the value of the source index map at a pixel, we can tell whether a source is there or not, as well as which source it is.
    ###
    ###     pse_metadata:                       The source metadata array, which stores the coordinates, brightness, and background of every found source from the PSE (up to nrefinepts).           
    ###     allintrmpoints_view:            An array holding the projected coordinates ("intermediate coordinates") of a set of sky coordinates downloaded from the catalogue. The success of the WCS is contingent on the geometric
    ###                                     correspondence between the intermediate coordinates and the pse coordinates.
    ###     psepoints_view:                 An array holding the pixel points of the npts brightest sources from the pse_metadata. The "view" suffix indicates that it is a "memory view", a type of high-efficency array
    ###                                     pointer datatype in Cython.
    ###     intrmpoints_view:               The equivalent of the above except for holding intermediate points.                 
    ###     S                               The scale parameter found by the optimizer when finding the initial solution in the WCS.
    ###     phi                             The rotation parameter found by the optimizer when finding the initial solution in the WCS.
    ###     CRPIXx                          The reference pixel location's x coordinate, also found by the optimizer in the WCS. (Once for the initial solution, and then improved
    ###                                     in the refinement stage). CRPIX is the point in the image distinguished by the property that vectors drawn from it to the psecoords only need
    ###                                     to undergo a rotation and a scaling (no translation) in order to land on the corresponding intermediate coords.
    ###     CRPIXy                          The reference pixel location's y coordinate.
    ###     CD1_1                           CD matrix element that equals S*cos(phi). All elements are found when refining the initial solution in the WCS. The complete CD matrix, together with CRPIX, is the final solution.
    ###     CD1_2                           CD matrix element that equals -S*sin(phi).
    ###     CD2_1                           CD matrix element that equals S*sin(phi).
    ###     CD2_2                           CD matrix element that equals S*cos(phi).
    ###
    ################################################################################################

    #Section 2
    print("\n")

    #Initialize some variables
    img_xmax = int(img.shape[1])
    img_ymax = int(img.shape[0])
    radius = scale*img_xmax/2 + buffer #degrees
    pixelradius = radius/scale  #pixels

    srcindexmap_initial = -1*np.ones(img.shape,dtype=int)
    srcindexmap_refine = -1*np.ones(img.shape,dtype=int)
    pse_metadata = np.zeros((nrefinepts,3),dtype=np.double)
    pse_metadata_inv = np.zeros((nrefinepts,3),dtype=np.double)


    num_psesources = PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, pixelradius, shape, srcindexmap_initial, srcindexmap_refine, pse_metadata, debug_report, filepath, verbosity, debug, printconsole=("Extracting sources from image...","done",verbosity,1))

    #Section 3
    print("\n")

    #Set up some more variables
    allintrmpoints = np.zeros((nrefinepts,2))
    catalogue_points = np.zeros((nrefinepts,3))
    mean_catcoords = np.zeros(2)

    num_catsources = getIntermediateCoords(ra, dec, scale, img_xmax, img_ymax, shape, filter, catalogue, pmepoch, nrefinepts, allintrmpoints, catalogue_points, mean_catcoords, gaiaqueries, debug_report, verbosity, debug, printconsole=("Getting intermediate coordinates...","done",verbosity,1))

    #Section 4
    print("\n")

    #And yet more variables
    pse_metadata_inv[:,0] = img_xmax - pse_metadata[:,0]
    pse_metadata_inv[:,1] = img_ymax - pse_metadata[:,1]
    pse_metadata_inv[:,2] = pse_metadata[:,2]
    matchdata = np.zeros((num_catsources,10))
    num_matches = np.zeros(1,dtype=int)

    if debug:
        np.savetxt(debug_report/"pse_metadata_inv.csv", pse_metadata_inv, delimiter=",")

    if nmatchpoints < nmatchpercent*npts:
        minmatches = nmatchpoints
    else:
        minmatches = nmatchpercent*npts
    kerneldiam = kernelrad*2+1

    headervals = np.zeros(26)

    WCS(scale, scalebnds, rotation, rotationbnds, npts, nrefinepts, vertextol, allintrmpoints, catalogue_points, mean_catcoords, pse_metadata, pse_metadata_inv, matchdata, num_matches, srcindexmap_initial, srcindexmap_refine, img_xmax, img_ymax, minmatches, kerneldiam, num_psesources, num_catsources, headervals, wcsdiagnostics, debug_report, filepath, user_dir, filename_body, verbosity, debug, printconsole=("Performing WCS...","done",verbosity,1))

    #Section 5
    print("\n")

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
    header['CCVALD1'] = (headervals[12], 'WCS field center on axis 1 (degrees)')
    header['CCVALD2'] = (headervals[13], 'WCS field center on axis 2 (degrees)')
    header['CCVALS1'] = ('{:d}h:{:d}m:{:.2f}s'.format(int(headervals[14]),int(headervals[15]),headervals[16]), 'WCS field center on axis 1 (sexigesimal h m s)')
    header['CCVALS2'] = ('{:d}d:{:d}am:{:.2f}as'.format(int(headervals[17]),int(headervals[18]),headervals[19]), 'WCS field center on axis 2 (sexigesimal d am as')
    header['CVAL1RM'] = (headervals[20], 'Mean of WCS residuals on axis 1 (arcsec)')
    header['CVAL1RS'] = (headervals[21], 'Standard dev of WCS residuals on axis 1 (arcsec')
    header['CVAL2RM'] = (headervals[22], 'Mean of WCS residuals on axis 2 (arcsec)')
    header['CVAL2RS'] = (headervals[23], 'Standard dev of WCS residuals on axis 2 (arcsec')
    header['CVALRM'] = (headervals[24], 'Mean of WCS residuals (arcsec)')
    header['CVALRS'] = (headervals[25], 'Standard dev of WCS residuals (arcsec)')

    outfilename = filename_body+' WCS'+filename_ext
    diagnostic = filename_body+' WCS diagnostic.csv'
    numbered_outfilename, numbered_diagnostic = insertCopyNumber(outfilename, diagnostic, filename_body, filename_ext)
    outfilepath = user_dir/numbered_outfilename
    diagnosticpath = user_dir/numbered_diagnostic
    writeto(outfilepath, img, header, overwrite=True, output_verify='silentfix', printconsole=("Writing to {}...".format(numbered_outfilename),"done",verbosity,1))

    if wcsdiagnostics is True:
        matches = num_matches[0]
        hdr = "PSE_XPIX, PSE_YPIX, CAT_RA, CAT_DEC, WCS_RADELTA, WCS_DECDELTA"
        savetxt(diagnosticpath, np.column_stack((matchdata[:matches,0], matchdata[:matches,1], matchdata[:matches,6]*180/pi, matchdata[:matches,7]*180/pi, matchdata[:matches,8], matchdata[:matches,9])), delimiter=',', header=hdr, comments='', printconsole=("Creating {}...".format(numbered_diagnostic),"done",verbosity,1))

    #The End
    print("\n")
    print("fastrometry completed successfully.")
    return

def callFromCommandLine():
    args = getCMDargs()
    solution = findWCS(args.filename, ra=args.ra, dec=args.dec, scale=args.scale, scalebnds=args.scalebnds, rotation=args.rotation, rotationbnds=args.rotationbnds, buffer=args.buffer, shape=args.fieldshape, catalogue=args.catalogue, filter=args.filter, pmepoch=args.pmepoch, npts=args.npts, nrefinepts=args.nrefinepts, pixsat=args.pixsat, kernelrad=args.kernelrad, sourcesep=args.sourcesep, vertextol=args.vertextol, nmatchpoints=args.nmatchpoints, nmatchpercent=args.nmatchpercent, wcsdiagnostics=args.wcsdiagnostics, save=args.save, load=args.load, verbosity=args.verbosity, debug=args.debug)
    #as of right now, solution is None -- you will be fine just calling findWCS in-place
    