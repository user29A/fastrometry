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

################
VERSION="1.0.9"
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
    parser.add_argument('-ra', required=True, help=formatting('The approximate right-ascension of the field center of the image. This can be supplied in either right-ascension sexagesimal (HH:MM:SS.S) format, degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either right-ascension sexagesimal or degree.decimal format. If in sexagesimal format, -ra must be passed as a string (surrounded with quotes).',80,4))
    parser.add_argument('-dec', required=True, help=formatting('The approximate declination of the field center of the image. This can be supplied in either declination sexagesimal format (DD:AM:AS.AS), degree.decimal format, or as the keyword in the FITS file which contains the relevant value in either declination sexagesimal or degree.decimal format. If in sexagesimal format, -dec must be passed as a string (surrounded with quotes). WARNING: If supplied in sexagesimal format, negative declinations must be preceded with a space (or else the argument parser assumes a new option).',80,4))
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
    parser.add_argument('--wcsdiagnostics', action='store_true', help=formatting('If this option is provided, fastrometry will write an additional csv file beside the FITS image file which provides diagnostic information about the WCS solution. The PSE centroids, their sky coordinate values via the WCS solution, the corresponding sky coordinates from the catalogue, and then the differentials, are written to the csv diagnostic file.',80,4))
    parser.add_argument('--overwrite', action='store_true', help=formatting('When set, will update the header WCS keywords of the existing supplied FITS file name and overwrite the file with the new header and existing image extension, instead of writing a new file.',80,4))
    parser.add_argument('--save', help=formatting('Save the options settings as a file to disk to recall later. User supplies an ID, for example: "--save UVIT". Useful if frequently processing different telescopic fields, allowing quick input of WCS solver settings. Saves all settings EXCEPT for filename, ra, dec, pmepoch, so that different images but taken from the same telescope can be processed at their unique individual characteristics.',80,4))
    parser.add_argument('--load', help=formatting('Load an options settings configuration. User supplies an ID that matches that of a previously-saved file. Useful when frequently processing the same telescopic field.',80,4))
    parser.add_argument('--version', '-V', action='version', version='fastrometry {}'.format(VERSION), help=formatting('Shows the current version of fastrometry.',80,4))
    parser.add_argument('--verbosity', help=formatting('Sets the verbosity level of print messages. A value of 1 represents normal verbosity and is the default. Set to 0 for a silent run, or 2 for higher verbosity.',80,4))
    parser.add_argument('--debug', action='store_true', help=formatting('Enters debug mode. Steps through the program, pausing at intervals to print diagnostic and debugging info to the print and display various plots and images. A debug report will be created and saved in the current working directory. --verbosity is automatically set to 2.',80,4))
    parser.add_argument('--h', '--help', '-h', '-help', action='help', default=argparse.SUPPRESS, help=formatting('Shows this help message.',80,4))

    input_args = parser.parse_args()

    return input_args

def drawOptionsTable(filename, ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics, overwrite, save, load, verbosity, debug):
    if verbosity == 2:
        if wcsdiagnostics is True:
            wcsdiagnostics_str = "Yes"
        else:
            wcsdiagnostics_str = "No"
        
        if overwrite is True:
            overwrite_str = "Yes"
        else:
            overwrite_str = "No"

        if save is None:
            save_str = ""
        else:
            save_str = str(save)

        if load is None:
            load_str = ""
        else:
            load_str = str(load)

        verbosity_str = str(verbosity)

        if debug is True:
            debug_str = "Yes"
        else:
            debug_str = "No"

        filename_str = str(filename)
        ra_str = str(ra)+" degrees"
        dec_str = str(dec)+" degrees"
        scale_str = str(scale)+" arcseconds/pixel"
        scalebnds_str = str(scalebnds)+"%"

        if rotation is None:
            rotation_str = "TBD"
        else:
            rotation_str = str(rotation)+" degrees"
        
        if rotationbnds is None:
            rotationbnds_str = ""
        else:
            rotationbnds_str = str(rotationbnds)+" degrees"
        
        buffer_str = str(buffer)+" arcminutes"
        fieldshape_str = str(fieldshape)
        catalogue_str = str(catalogue)
        filter_str = str(filter)

        if pmepoch == 0:
            pmepoch_str = "0 (off)"
        else:
            pmepoch_str = str(pmepoch)

        npts_str = str(npts)+" points"
        nrefinepts_str = str(nrefinepts)+" points"
        
        if pixsat == 0:
            pixsat_str = "0 (off)"
        else:
            pixsat_str = str(pixsat)

        kernelrad_str = str(kernelrad)+" pixels"
        sourcesep_str = str(sourcesep)+" pixels"
        vertextol_str = str(vertextol)+" degrees"
        nmatchpoints_str = str(nmatchpoints)+" points"
        nmatchpercent_str = str(nmatchpercent)+"%"

        print("| ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '")
        print("| '                              ==Options==                              '")
        print("| '                                                                       '")
        print("| '  Settings:                                                            '")
        print("| '    wcsdiagnostics: {:51}'".format(wcsdiagnostics_str))
        print("| '    overwrite: {:56}'".format(overwrite_str))
        print("| '    save: {:61}'".format(save_str))
        print("| '    load: {:61}'".format(load_str))
        print("| '    verbosity: {:56}'".format(verbosity_str))
        print("| '    debug: {:60}'".format(debug_str))
        print("| '                                                                       '")
        print("| '  Parameters:                                                          '")
        print("| '    filename: {:57}'".format(filename_str))
        print("| '    ra: {:63}'".format(ra_str))
        print("| '    dec: {:62}'".format(dec_str))
        print("| '    scale: {:60}'".format(scale_str))
        print("| '    scalebnds: {:56}'".format(scalebnds_str))
        print("| '    rotation: {:57}'".format(rotation_str))
        print("| '    rotationbnds: {:53}'".format(rotationbnds_str))
        print("| '    buffer: {:59}'".format(buffer_str))
        print("| '    fieldshape: {:55}'".format(fieldshape_str))
        print("| '    catalogue: {:56}'".format(catalogue_str))
        print("| '    filter: {:59}'".format(filter_str))
        print("| '    pmepoch: {:58}'".format(pmepoch_str))
        print("| '    npts: {:61}'".format(npts_str))
        print("| '    nrefinepts: {:55}'".format(nrefinepts_str))
        print("| '    pixsat: {:59}'".format(pixsat_str))
        print("| '    kernelrad: {:56}'".format(kernelrad_str))
        print("| '    sourcesep: {:56}'".format(sourcesep_str))
        print("| '    vertextol: {:56}'".format(vertextol_str))
        print("| '    nmatchpoints: {:53}'".format(nmatchpoints_str))
        print("| '    nmatchpercent: {:52}'".format(nmatchpercent_str))
        print("| '                                                                       '")
        print("| ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '")

def saveRemainingOptions(save, user_dir, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, verbosity):
    if verbosity >= 1:
        print("Saving options...")
    try:
        #save = re.sub(r'[^\w\-_\. ]', '', save)
        contents = {
            'scale' : scale,
            'scalebnds' : scalebnds,
            'rotation' : rotation,
            'rotationbnds' : rotationbnds,
            'buffer' : buffer,
            'fieldshape' : fieldshape,
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
        jobject = json.dumps(contents)
        if not (user_dir/'saved_options').is_dir():
            (user_dir/'saved_options').mkdir()
        with open(user_dir/'saved_options'/'{}.json'.format(save),'w') as fp:
            fp.write(jobject)
    except Exception as e:
        print(e)
        sys.exit("ERROR: Could not save options.")      
    if verbosity >= 1:
        print("done")
    return save    

def loadRemainingOptions(load, user_dir, verbosity):
    if verbosity >= 1:
        print("Loading options...")
    try:
        with open(user_dir/'saved_options'/'{}.json'.format(load),'r') as fp:
            jobject = json.load(fp)
            scale = jobject['scale']
            scalebnds = jobject['scalebnds']
            rotation = jobject['rotation']
            rotationbnds = jobject['rotationbnds']
            buffer = jobject['buffer']
            fieldshape = jobject['fieldshape']
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
    except Exception as e:
        print(e)
        sys.exit("ERROR: Could not load options. Check that a json file exists with the specified id.")
    if verbosity >= 1:
        print("done")
    return scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent

def validateOptions(filename, ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics, overwrite, save, load, verbosity, debug):

    if verbosity != 0 and verbosity != "0":
        print("Validating options...")

    if verbosity != 0 and verbosity != "0":
        print("| Validating --verbosity...")
    if verbosity is None:
        verbosity = 1
    else:
        try:
            verbosity = int(verbosity)
        except:
            sys.exit("ERROR: --verbosity must be an integer.")
        if not 0 <= verbosity <= 2:
            sys.exit("ERROR: --verbosity must be between 0 and 2.")
    if verbosity >= 1:
        print("| done")

    if verbosity >= 1:
        print("| Validating -debug...")
    if debug is None:
        debug = False
        if verbosity == 2:
            print("| | --debug not supplied. Defaulting to False.")
    else:
        try:
            debug = bool(debug)
        except:
            sys.exit("ERROR: --debug must be a boolean.")
        if debug is True:
            verbosity = 2
    if verbosity >= 1:
        print("| done")

    if verbosity >= 1:
        print("| Validating --overwrite...")
    if overwrite is None:
        overwrite = False
        if verbosity == 2:
            print("| | --overwrite not supplied. Defaulting to False.")
    else:
        try:
            overwrite = bool(overwrite)
        except:
            sys.exit("ERROR: --overwrite must be a boolean.")
    if verbosity >= 1:
        print("| done")

    if verbosity >= 1:
        print("| Validating --wcsdiagnostics...")
    if wcsdiagnostics is None:
        wcsdiagnostics = False
        if verbosity == 2:
            print("| | --wcsdiagnostics not supplied. Defaulting to False.")
    else:
        try:
            wcsdiagnostics = bool(wcsdiagnostics)
        except:
            sys.exit("ERROR: --wcsdiagnostics must be a boolean.")
    if verbosity >= 1:
        print("| done")

    if verbosity >= 1:
        print("| Validating -filename...")
    filename = Path(filename)
    filepath = filename.resolve()
    if not filepath.is_file():
        sys.exit("ERROR: could not resolve the specified FITS file. Check that you are in the directory containing your FITS files.")
    user_dir = filepath.parents[0]
    filename_body = filename.stem
    filename_ext = filename.suffix
    with fits.open(filepath) as hdul:
        img = hdul[0].data.astype(np.double)
        header = hdul[0].header
    if verbosity >= 1:
        print("| done")

    if verbosity >= 1:
        print("| Validating -ra...")
    try:
        ra = float(ra)
        if not 0 <= ra <= 360:  
            sys.exit("ERROR: -ra must be between 0 and 360 (assuming -ra was given in degree.decimal format).")
        if verbosity == 2:
            print("| | Interpreting -ra in degree.decimal format.")
    except:
        parts = ra.split(':')
        if len(parts) == 1:
            if ra not in header:
                sys.exit("ERROR: could not find the specified keyword in the FITS header (assuming -ra was supplied as a keyword).")
            if verbosity == 2:
                print("| | Found {} keyword in header.".format(ra))
            try:
                ra = float(header[ra])
            except:
                parts = header[ra].split(':')
                hrs = parts[0]
                mins = parts[1]
                secs = parts[2]
                try:
                    hrs = int(hrs)
                except:
                    sys.exit("ERROR: hours must be an integer (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
                try:
                    mins = int(mins)
                except:
                    sys.exit("ERROR: minutes must be an integer (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
                try:
                    secs = float(secs)
                except:
                    sys.exit("ERROR: seconds must be a number (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
                if not 0 <= hrs <= 24:
                    sys.exit("ERROR: hours must be between 0 and 24 (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
                if not 0 <= mins <= 60:
                    sys.exit("ERROR: minutes must be between 0 and 60 (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
                if not 0 <= secs <= 60: 
                    sys.exit("ERROR: seconds must be between 0 and 60 (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
                ra = float(hrs)*15+float(mins)*15/60+secs*15/3600
                if verbosity == 2:
                    print("| | Interpreting -ra in sexagesimal format.")
        elif len(parts) == 3:
            hrs = parts[0]
            mins = parts[1]
            secs = parts[2]
            try:
                hrs = int(hrs)
            except:
                sys.exit("ERROR: hours must be an integer (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
            try:
                mins = int(mins)
            except:
                sys.exit("ERROR: minutes must be an integer (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
            try:
                secs = float(secs)
            except:
                sys.exit("ERROR: seconds must be a number (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
            if not 0 <= hrs <= 24:
                sys.exit("ERROR: hours must be between 0 and 24 (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
            if not 0 <= mins <= 60:
                sys.exit("ERROR: minutes must be between 0 and 60 (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
            if not 0 <= secs <= 60: 
                sys.exit("ERROR: seconds must be between 0 and 60 (assuming -ra was given in sexagesimal format HH:MM:SS.S).")
            ra = float(hrs)*15+float(mins)*15/60+secs*15/3600
            if verbosity == 2:
                print("| | Interpreting -ra in sexagesimal format.")
        if verbosity >= 1:
            print("| done")

    if verbosity >= 1:
        print("| Validating -dec...")
    try:
        dec = float(dec)
        if not -90 <= dec <= 90:  
            sys.exit("ERROR: -dec must be between 0 and 360 (assuming -dec was given in degree.decimal format).")
        if verbosity == 2:
            print("| | Interpreting -dec in degree.decimal format.")
    except:
        parts = dec.split(':')
        if len(parts) == 1:
            if dec not in header:
                sys.exit("ERROR: could not find the specified keyword in the FITS header (assuming -dec was supplied as a keyword).")
            if verbosity == 2:
                print("| | Found {} keyword in header.".format(dec))
            try:
                dec = float(header[dec])
            except:
                parts = header[dec].split(":")
                deg = parts[0]
                amins = parts[1]
                asecs = parts[2]
                try:
                    deg = int(deg)
                except:
                    sys.exit("ERROR: degrees must be an integer (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
                try:
                    amins = int(amins)
                except:
                    sys.exit("ERROR: arcminutes must be an integer (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
                try:
                    asecs = float(asecs)
                except:
                    sys.exit("ERROR: arcseconds must be a number (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
                if not -90 <= deg <= 90:
                    sys.exit("ERROR: degrees must be between -90 and 90 (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
                if not 0 <= amins <= 60:
                    sys.exit("ERROR: arcminutes must be between 0 and 60 (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
                if not 0 <= asecs <= 60: 
                    sys.exit("ERROR: arcseconds must be between 0 and 60 (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
                if deg >= 0:
                    dec = float(deg)+float(amins)/60+asecs/3600
                else:
                    dec = float(deg)-float(amins)/60-asecs/3600
                if verbosity == 2:
                    print("| | Interpreting -dec in sexagesimal format.")
        elif len(parts) == 3:
            deg = parts[0]
            amins = parts[1]
            asecs = parts[2]
            try:
                deg = int(deg)
            except:
                sys.exit("ERROR: degrees must be an integer (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
            try:
                amins = int(amins)
            except:
                sys.exit("ERROR: arcminutes must be an integer (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
            try:
                asecs = float(asecs)
            except:
                sys.exit("ERROR: arcseconds must be a number (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
            if not -90 <= deg <= 90:
                sys.exit("ERROR: degrees must be between -90 and 90 (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
            if not 0 <= amins <= 60:
                sys.exit("ERROR: arcminutes must be between 0 and 60 (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
            if not 0 <= asecs <= 60: 
                sys.exit("ERROR: arcseconds must be between 0 and 60 (assuming -dec was given in sexagesimal format DD:AM:AS.AS).")
            if deg >= 0:
                dec = float(deg)+float(amins)/60+asecs/3600
            else:
                dec = float(deg)-float(amins)/60-asecs/3600
            if verbosity == 2:
                print("| | Interpreting -dec in sexagesimal format.")
    if verbosity >= 1:
            print("| done")

    if verbosity >= 1:
        print("| Validating -pmepoch...")
    if pmepoch is None:
        pmepoch = 0
        if verbosity == 2:
            print("| | -pmepoch not supplied. Defaulting to 0.")
    else:
        try:
            pmepoch = float(pmepoch)
        except:
            sys.exit("ERROR: -pmepoch must be a number.")
        if not 1950 <= pmepoch:
            sys.exit("ERROR: -pmepoch must be greater than or equal to 1950.")
    if verbosity >= 1:
        print("| done")

    if load is not None:
        scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent = loadRemainingOptions(load, user_dir, verbosity)
    else:
        if verbosity >= 1:
            print("| Validating -scale...")
        if scale is None:
            sys.exit('ERROR: -scale is required.')          
        try:
            scale = float(scale)        
        except:
            sys.exit('ERROR: -scale must be a number.')
        if not scale > 0:
            sys.exit('ERROR: -scale must be greater than 0.')
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -scalebnds...")
        if scalebnds is None:
            scalebnds = 5
            if verbosity == 2:
                print("| | -scalebnds not supplied. Defaulting to 5.")
        else:
            try:
                scalebnds = float(scalebnds)                 
            except:
                sys.exit('ERROR: -scalebnds must be a number.')
            if not scalebnds >= 0:
                sys.exit('ERROR: -scalebnds must be greater than 0.') 
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -rotation and -rotationbnds...")
        if rotation is None and rotationbnds is None:
            if verbosity == 2:
                print("| | neither -rotation nor -rotationbnds supplied. Will be guessed in the WCS solver.")
        elif rotation is None and rotationbnds is not None:
            try:
                rotationbnds = float(rotationbnds)           
            except:
                sys.exit('ERROR: -rotationbnds must be a number.')             
            if not 0 <= rotationbnds <= 180:
                sys.exit('ERROR: -rotationbnds must be between 0 and 180.')
            rotation = 0     
            if verbosity == 2:
                print("| | -rotation not supplied. Defaulting to 0.")
        elif rotation is not None and rotationbnds is None:
            try:
                rotation = float(rotation)             
            except:
                sys.exit('ERROR: -rotation must be a number.')
            if not -180 <= rotation < 180:
                sys.exit('ERROR: -rotation must be between -180 and 180.')
            rotationbnds = 10+vertextol
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
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -buffer...")
        if buffer is None:
            buffer = 0
            if verbosity == 2:
                print("| | -buffer not supplied. Defaulting to 0.")
        else:
            try:
                buffer = float(buffer)
            except:
                sys.exit("ERROR: -buffer must be a number.")
        if verbosity >= 1:
            print("| done...")

        if verbosity >= 1:
            print("| Validating -fieldshape...")
        if fieldshape is None:
            fieldshape = 'rectangle'
            if verbosity == 2:
                print("| | -fieldshape not supplied. Defaulting to 'rectangle'.", 2, verbosity)
        else:
            if fieldshape not in ['rectangle','circle']:
                sys.exit("ERROR: -fieldshape must be from the following options: ['rectangle','circle']")
            if fieldshape == 'circle':
                if not header['NAXIS1'] == header['NAXIS2']:
                    sys.exit("ERROR: circle query requires a square image, i.e., NAXIS1 = NAXIS2.")
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -catalogue and -filter...")
        if catalogue is None:
            catalogue = 'GaiaDR3'
            if verbosity == 2:
                print("| | -catalogue not supplied. Defaulting to 'GaiaDR3'")
        if catalogue == 'GaiaDR3':
            if filter is None:
                filter = 'g'
                if verbosity == 2:
                    print("| | -filter not supplied. Defaulting to 'g'.")
            else:
                if filter not in ['bp','g','rp']:
                    sys.exit("ERROR: -filter for the GaiaDR3 catalogue must be from the following options: ['bp','g','rp'].")
        elif catalogue not in ['GaiaDR3']:
            sys.exit("ERROR: -catalogue must be from the following options: ['GaiaDR3'].")
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -npts...")
        if npts is None:
            npts = 75
            if verbosity == 2:
                print("| | -npts not supplied. Defaulting to 75.")
        else:
            try:
                npts = int(npts)
            except:
                sys.exit('ERROR: -npts must be an integer.')
            if not 10 <= npts <= 300:
                sys.exit("ERROR: -npts must be between 10 and 300.")
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -nrefinepts...")
        if nrefinepts is None:
            nrefinepts = 500
            if verbosity == 2:
                print("| | -nrefinepts not supplied. Defaulting to 500.")
        else:
            try:
                nrefinepts = int(nrefinepts)
            except:
                sys.exit('ERROR: -nrefinepts must be an integer.')
            if not npts <= nrefinepts <= 1000:
                sys.exit("ERROR: -nrefinepts must be between npts and 1000.")
        if verbosity >= 1:
            print("| done")
            
        if verbosity >= 1:
            print("| Validating -pixsat...")
        if pixsat is None:
            pixsat = 0.
            if verbosity == 2:
                print("| | -pixsat not supplied. Defaulting to 0 (off).")
        else:
            try:
                pixsat = float(pixsat)
            except:
                sys.exit('ERROR: -pixsat must be a number.')
            if not pixsat >= 0: 
                sys.exit("ERROR: -pixsat must be greater than or equal to 0.")
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -kernelrad...")
        if kernelrad is None:
            kernelrad = 2
            if verbosity == 2:
                print("| | -kernelrad not supplied. Defaulting to 2.")
        else:
            try:
                kernelrad = int(kernelrad)
            except:
                sys.exit("ERROR: -kernelrad must be an integer.")
            if not kernelrad >= 2:
                sys.exit("ERROR: -kernelrad must be greater than or equal to 2.")
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -sourcesep...")  
        if sourcesep is None:
            sourcesep = 25
            if verbosity == 2:
                print("| | -sourcesep not supplied. Defaulting to 25.")
        else:
            try:
                sourcesep = int(sourcesep)
            except:
                sys.exit("ERROR: -sourcesep must be an integer.")
            if not kernelrad <= sourcesep <= 45:
                sys.exit("ERROR: -sourcesep must be greater than or equal to -kernelrad and less than or equal to 45.")
        if verbosity >= 1:
            print("| done") 
        
        if verbosity >= 1:
            print("| Validating -vertextol...")
        if vertextol is None:
            vertextol = 0.25
            if verbosity == 2:
                print("| | -vertextol not supplied. Defaulting to 0.25.")
        else:
            try:
                vertextol = float(vertextol)
            except:
                sys.exit("ERROR: -vertextol must be a number.")
            if not 0 < vertextol < 2:
                sys.exit("ERROR: -vertextol must be between 0 and 2.")  
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating -nmatchpoints...")
        if nmatchpoints is None:
            nmatchpoints = 6
            if verbosity == 2:
                print("| | -nmatchpoints not supplied. Defaulting to 6.")
        else:
            try:
                nmatchpoints = int(nmatchpoints)
            except:
                sys.exit("ERROR: -nmatchpoints must be an integer.")
            if not nmatchpoints > 2:
                sys.exit("ERROR: -nmatchpoints must be greater than 2.")
        if verbosity >= 1:
            print("| done")

        if verbosity >= 1:
            print("| Validating nmatchpercent...")
        if nmatchpercent is None:
            nmatchpercent = 25
            if verbosity == 2:
                print("| | -nmatchpercent not supplied. Defaulting to 25.")
        else:
            try:
                nmatchpercent = float(nmatchpercent)
            except:
                sys.exit("ERROR: -nmatchpercent must be a number.")
            if not nmatchpercent > 10:
                sys.exit("ERROR: -nmatchpercent must be greater than 10.")
        if verbosity >= 1:
            print("| done")

    if save is not None:
        save = saveRemainingOptions(save, user_dir, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, verbosity)

    if verbosity >= 1:
        print("done")
    
    return filepath, user_dir, filename_body, filename_ext, img, header, filename, ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics, overwrite, save, load, verbosity, debug

def setUpFolders(debug, user_dir, filename_body, verbosity):
    if verbosity >= 1:
        print("Setting up folders...")
    debug_folder = (user_dir/'debug')
    if not debug_folder.is_dir():
            debug_folder.mkdir()
    if debug:
        report_name = str(filename_body)+str(datetime.now().strftime("_%y-%m-%d_%H-%M-%S"))
        debug_report = (user_dir/'debug'/report_name)
        debug_report.mkdir()
    else:
        debug_report = None
    gaiaqueries = (user_dir/'gaiaqueries')
    if not gaiaqueries.is_dir():
        gaiaqueries.mkdir()
    if verbosity >= 1:
        print("done")
    return debug_folder, debug_report, gaiaqueries

def findWCS(filename, ra=None, dec=None, scale=None, scalebnds=None, rotation=None, rotationbnds=None, buffer=None, fieldshape=None, catalogue=None, filter=None,  pmepoch=None, npts=None, nrefinepts=None, pixsat=None, kernelrad=None, sourcesep=None, vertextol=None, nmatchpoints=None, nmatchpercent=None, wcsdiagnostics=None, overwrite=None, save=None, load=None, verbosity=None, debug=None):

    #Section 1
    print("\n")

    filepath, user_dir, filename_body, filename_ext, img, header, filename, ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics, overwrite, save, load, verbosity, debug = validateOptions(filename, ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics, overwrite, save, load, verbosity, debug)

    drawOptionsTable(filename, ra, dec, scale, scalebnds, rotation, rotationbnds, buffer, fieldshape, catalogue, filter, pmepoch, npts, nrefinepts, pixsat, kernelrad, sourcesep, vertextol, nmatchpoints, nmatchpercent, wcsdiagnostics, overwrite, save, load, verbosity, debug)
    
    debug_folder, debug_report, gaiaqueries = setUpFolders(debug, user_dir, filename_body, verbosity)

    #Section 2
    print("\n")

    img_xmax = int(img.shape[1])
    img_ymax = int(img.shape[0])
    radius = scale/3600*img_xmax/2 + buffer/60  #degrees
    pixelradius = radius/(scale/3600)           #pixels

    srcindexmap_initial = -1*np.ones(img.shape, dtype=int)
    srcindexmap_refine = -1*np.ones(img.shape, dtype=int)
    pse_metadata = np.zeros((nrefinepts,3), dtype=np.double)
    pse_metadata_inv = np.zeros((nrefinepts,3), dtype=np.double)

    num_psesources = PSE(img, img_xmax, img_ymax, kernelrad, sourcesep, pixsat, npts, nrefinepts, pixelradius, fieldshape, srcindexmap_initial, srcindexmap_refine, pse_metadata, debug_report, filepath, debug, verbosity)

    #Section 3
    print("\n")

    allintrmpoints = np.zeros((nrefinepts,2))
    catalogue_points = np.zeros((nrefinepts,3))
    mean_catcoords = np.zeros(2)

    num_catsources = getIntermediateCoords(ra, dec, scale, img_xmax, img_ymax, fieldshape, buffer, filter, catalogue, pmepoch, nrefinepts, allintrmpoints, catalogue_points, mean_catcoords, gaiaqueries, debug_report, overwrite, debug, verbosity)

    #Section 4
    print("\n")

    scale = scale/3600*pi/180   #from arcsec/pixel to radians/pixel
    scalebnds = scalebnds*.01   #from % to a decimal
    if rotation is not None:
        rotation = rotation*pi/180  #from degrees to radians
    if rotationbnds is not None:
        rotationbnds = rotationbnds*pi/180  #from degrees to radians
    vertextol = vertextol*pi/180    #from degrees to radians
    nmatchpercent = nmatchpercent*.01   #from % to a decimal

    if nmatchpoints < nmatchpercent*npts:
        minmatches = nmatchpoints
    else:
        minmatches = nmatchpercent*npts
    kerneldiam = kernelrad*2+1

    pse_metadata_inv[:,0] = img_xmax - pse_metadata[:,0]
    pse_metadata_inv[:,1] = img_ymax - pse_metadata[:,1]
    pse_metadata_inv[:,2] = pse_metadata[:,2]
    matchdata = np.zeros((num_catsources,10))
    num_matches = np.zeros(1,dtype=int)

    if debug:
        np.savetxt(debug_report/"pse_metadata_inv.csv", pse_metadata_inv, delimiter=",")

    headervals = np.zeros(26)

    WCS(scale, scalebnds, rotation, rotationbnds, npts, nrefinepts, vertextol, allintrmpoints, catalogue_points, mean_catcoords, pse_metadata, pse_metadata_inv, matchdata, num_matches, srcindexmap_initial, srcindexmap_refine, img_xmax, img_ymax, minmatches, kerneldiam, num_psesources, num_catsources, headervals, wcsdiagnostics, debug_report, filepath, user_dir, filename_body, debug, verbosity)

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
    header['CCVALS1'] = ('{:d}h:{:d}m:{:.2f}s'.format(int(headervals[14]),int(headervals[15]),headervals[16]), 'WCS field center on axis 1 (sexagesimal h m s)')
    header['CCVALS2'] = ('{:d}d:{:d}am:{:.2f}as'.format(int(headervals[17]),int(headervals[18]),headervals[19]), 'WCS field center on axis 2 (sexagesimal d am as')
    header['CVAL1RM'] = (headervals[20], 'Mean of WCS residuals on axis 1 (arcsec)')
    header['CVAL1RS'] = (headervals[21], 'Standard dev of WCS residuals on axis 1 (arcsec')
    header['CVAL2RM'] = (headervals[22], 'Mean of WCS residuals on axis 2 (arcsec)')
    header['CVAL2RS'] = (headervals[23], 'Standard dev of WCS residuals on axis 2 (arcsec')
    header['CVALRM'] = (headervals[24], 'Mean of WCS residuals (arcsec)')
    header['CVALRS'] = (headervals[25], 'Standard dev of WCS residuals (arcsec)')

    if overwrite:
        numbered_outfilename = filename_body+filename_ext
        numbered_diagnostic = filename_body+'.csv'
    else:
        outfilename = filename_body+' WCS'+filename_ext
        diagnostic = filename_body+' WCS diagnostic.csv'
        numbered_outfilename, numbered_diagnostic = insertCopyNumber(outfilename, diagnostic, filename_body, filename_ext)
    outfilepath = user_dir/numbered_outfilename
    diagnosticpath = user_dir/numbered_diagnostic
    if verbosity >= 1:
        print("Writing to {}...".format(numbered_outfilename))
    fits.writeto(outfilepath, img, header, overwrite=True, output_verify='silentfix')
    if verbosity >= 1:
        print("done")

    if wcsdiagnostics:
        matches = num_matches[0]
        hdr = "PSE_XPIX, PSE_YPIX, CAT_RA, CAT_DEC, WCS_RADELTA, WCS_DECDELTA"
        if verbosity >= 1:
            print("Writing to {}...".format(numbered_diagnostic))
        np.savetxt(diagnosticpath, np.column_stack((matchdata[:matches,3], matchdata[:matches,4], matchdata[:matches,6]*180/pi, matchdata[:matches,7]*180/pi, matchdata[:matches,8], matchdata[:matches,9])), delimiter=',', header=hdr, comments='')
        if verbosity >= 1:
            print("done")

    #The End
    print("\n")
    print("fastrometry completed successfully.")
    return

def callFromCommandLine():
    args = getCMDargs()
    solution = findWCS(args.filename, ra=args.ra, dec=args.dec, scale=args.scale, scalebnds=args.scalebnds, rotation=args.rotation, rotationbnds=args.rotationbnds, buffer=args.buffer, fieldshape=args.fieldshape, catalogue=args.catalogue, filter=args.filter, pmepoch=args.pmepoch, npts=args.npts, nrefinepts=args.nrefinepts, pixsat=args.pixsat, kernelrad=args.kernelrad, sourcesep=args.sourcesep, vertextol=args.vertextol, nmatchpoints=args.nmatchpoints, nmatchpercent=args.nmatchpercent, wcsdiagnostics=args.wcsdiagnostics, overwrite=args.overwrite, save=args.save, load=args.load, verbosity=args.verbosity, debug=args.debug)
    #as of right now, solution is None
    