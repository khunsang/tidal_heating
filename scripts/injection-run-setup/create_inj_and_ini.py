#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : create_inj_and_ini.py
# Author            : Khun Sang Phukon <khunsang@gmail.com>
# Date              : 17.09.2020
# Last Modified Date: 17.09.2020
# Last Modified By  : Khun Sang Phukon <khunsang@gmail.com>
from optparse import OptionParser
#from ligo.lw import ligolw
#from ligo.lw import lsctables
from glue.lal import Cache
#from ligo.lw import utils as ligolw_utils
from glue.ligolw import utils as ligolw_utils
from glue.ligolw import ligolw, table, lsctables
from astropy.units.si import sday
from bilby.gw import conversion
import os
import sys
import numpy as np
import lalsimulation as lalsim
import pandas as pd
import lal

@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass

def gmst_rad_from_geocetric_end_time(gps_time):
	phase = (float(gps_time) / sday.si.scale ) * 2 * np.pi
	gmst_rad = phase % (2*np.pi)
	return gmst_rad

def logitude_to_RA_no_lalsuite(longitude, gps_time):
	gmst = gmst_rad_from_geocetric_end_time(gps_time)
	ra = longitude + gmst
	return ra % (2*np.pi)

def tau0_from_mass1_mass2(mass1, mass2, f_lower):
	mtotal = mass1 + mass2
	eta = mass1 * mass2/mtotal**2
	numerator = 5. / (256. * ( np.pi * f_lower)**(8./3.))
	tau0 = numerator /	((mtotal * lal.MTSUN_SI)**(5./3.) * eta)
	return tau0


def mass_params( mass1, mass2):
	
	if isinstance( mass1, np.ndarray) and isinstance( mass2, np.ndarray):
		
		m1 = np.empty_like(mass1)
		m2 = np.empty_like(mass2)

		for i, (m1val, m2val) in enumerate(zip(mass1, mass2)):
			if m1val >= m2val:
				m1[i] = m1val
				m2[i] = m2val
			else:
				m1[i] = m2val
				m2[i] = m1val
		
		q = conversion.component_masses_to_mass_ratio ( m1, m2 )
		chirp_mass = conversion.component_masses_to_chirp_mass( mass1, mass2 )

	return chirp_mass, q, m1, m2


def mkdir(dirname):
	if not os.path.exists(dirname):
		try:
			os.makedir(dirname)
		except:
			raise OSError("Directroy exists. Can't create destination directory (%s)!" % (dirname)) 


def get_h_params( mass1, mass2, chi1, chi2, H1, H2 ):
	'''
	Return: Heff5, Heff8
	'''
	mtotal = mass1 + mass2
	A1 = H1*(mass1**3/mtotal**3)*chi1*(3*chi1**2 + 1)
	A2 = H2*(mass2**3/mtotal**3)*chi2*(3*chi2**2 + 1)
	H5 = A1 + A2
	
	term1 = H1*( mass1/mtotal )**4*( 3*chi1**2 + 1 )*( np.sqrt(1-chi1**2) + 1)
	term2 = H2*( mass2/mtotal )**4*( 3*chi2**2 + 1 )*( np.sqrt(1-chi2**2) + 1)
	H8 = 4*np.pi*H5 + term1 + term2
	return H5, H8


def get_max_mass_eos( eos_name='MS1'  ):
	if eos_name==None:
		print ("EOS name is empty, pass a EOS name from the following")
		print (lalsim.__dict__['SimNeutronStarEOSNames'])
		return None

	try:
		assert eos_name in list(lalsim.__dict__['SimNeutronStarEOSNames'])

	except AssertionError:
		print ("eos_name {} is not available in lalsimulation".format(eos_name))
		print ('Allowed EOS are: ', lalsim.__dict__['SimNeutronStarEOSNames'])
		sys.exit(0)

	eos = lalsim.SimNeutronStarEOSByName(eos_name)
	eos_family = lalsim.CreateSimNeutronStarFamily(eos)
	return lalsim.SimNeutronStarMaximumMass(eos_family)/lal.MSUN_SI


def get_lambda1_lambda2_from_mass1_mass2 ( mass1, mass2, eos_name='MS1' ):

	if eos_name==None:
		print ("EOS name is empty, pass a EOS name from the following")
		print (lalsim.__dict__['SimNeutronStarEOSNames'])
		return None
	
	try:
		assert eos_name in list(lalsim.__dict__['SimNeutronStarEOSNames'])
	
	except AssertionError:
		print ("eos_name {} is not available in lalsimulation".format(eos_name))
		print ('Allowed EOS are: ', lalsim.__dict__['SimNeutronStarEOSNames'])
		sys.exit(0)

	eos = lalsim.SimNeutronStarEOSByName(eos_name)
	eos_family = lalsim.CreateSimNeutronStarFamily(eos)

	max_mass = lalsim.SimNeutronStarMaximumMass(eos_family)/lal.MSUN_SI

	## lambda values based on max mass criterion

	if isinstance( mass1, np.ndarray) and isinstance( mass2, np.ndarray):
		
		lambda1 = np.empty_like(mass1)
		lambda2 = np.empty_like(mass2)
		
		for i, (m1val, m2val) in enumerate(zip(mass1, mass2)):
			
			if m1val <= max_mass:
				r1 = lalsim.SimNeutronStarRadius( m1val * lal.MSUN_SI, eos_family )
				# Kappa Dimensionless Love number
				Kappa2_1 = lalsim.SimNeutronStarLoveNumberK2( m1val * lal.MSUN_SI, eos_family )
				# Compactness
				C1 = ( m1val * lal.MSUN_SI * lal.G_SI ) / (  r1 * lal.C_SI**2	)
				# Tidal deformability
				lambda1[i] = 2./3 * Kappa2_1 / C1**5
			else:
				lambda1[i] = 0
			
			if m2val <= max_mass:
				r2 = lalsim.SimNeutronStarRadius( m2val * lal.MSUN_SI, eos_family )
				# Kappa Dimensionless Love number
				Kappa2_2 = lalsim.SimNeutronStarLoveNumberK2( m2val * lal.MSUN_SI, eos_family )
				# Compactness
				C2 = ( m2val * lal.MSUN_SI * lal.G_SI ) / (  r2 * lal.C_SI**2	)
				# Tidal deformability
				lambda2[i] = 2./3 * Kappa2_2 / C2**5
			else:
				lambda2[i] = 0
	return lambda1, lambda2


#lalapps_chirplen --m1 11 --m2 5 --flow 30

def parse_command_line():
	parser = OptionParser(description = __doc__)
	parser.add_option("--injection-file", metavar = "filename", default= "full_inj.xml.gz", help = "Set the injection xml file.")
	parser.add_option("--eos", metavar = "eos name", default="AP3", help = "EOS for BNS params.")
	parser.add_option("--flow", metavar = "flow Hz", default="20", help = "EOS for BNS params.", type='float')
	parser.add_option("--inj-dir", metavar = "injection directory", default="inj", help = "Set the injection directory")
	options, filenames = parser.parse_args()
	return options, filenames

options, filenames = parse_command_line()



inj_dir = options.inj_dir

mkdir(inj_dir)

inj_xml_file = os.path.join( inj_dir, options.injection_file)

cmd_injection_file = 'lalapps_inspinj --m-distr componentMass --min-mass1 1.0 --max-mass1 5.0 \
		--min-mass2 1.0 --max-mass2 5.0 --min-mtotal 2.0 --max-mtotal 10.0\
		--enable-spin --aligned  --min-spin1 0	--min-spin2 0  --max-spin1 0.75 --max-spin2 0.75\
		--i-distr uniform --l-distr random --d-distr volume --min-distance 40000 --max-distance 250000\
		--seed 100 --t-distr uniform --time-step 2000 --time-interval 25 --gps-start-time 1126051217  --gps-end-time 1128299417\
		--f-lower {} --waveform TaylorF2 --output {}'.format( options.flow, inj_xml_file )

#os.system('lalapps_inspinj --help')

os.system(cmd_injection_file)

#Read inj files sim_inspiral
xmldoc = ligolw_utils.load_filename( inj_xml_file, verbose = True, contenthandler = LIGOLWContentHandler)

xml_table = table.get_table(  xmldoc, lsctables.SimInspiralTable.tableName	)


mass1_samples = xml_table.get_column('mass1')
mass2_samples = xml_table.get_column('mass2')

mchirp_samples = xml_table.get_column('mchirp')
mass_ratio_samples = conversion.component_masses_to_mass_ratio(mass1_samples, mass2_samples)

geocent_end_time_samples =	xml_table.get_column('geocent_end_time')


longitude_samples = xml_table.get_column('longitude')
### make sure that longitudes are in the range [0,2\pi]

ra_samples = []

for i in range(len(geocent_end_time_samples)):
	ra = logitude_to_RA_no_lalsuite(  longitude_samples[i],
			geocent_end_time_samples[i] )
	ra_samples.append(ra)

ra_samples = np.asarray(ra_samples)
dec_samples = xml_table.get_column('latitude')
distance_samples = xml_table.get_column('distance')


ref_freq = 40.

phi_jl_array = []
tilt_1_array = []
tilt_2_array = []
phi_12_array = []
a_1_array = []
a_2_array = []

chi1_array = xml_table.get_column('spin1z')
chi2_array = xml_table.get_column('spin2z')


for i in range( len(ra_samples) ):
	param_array = [ float(a) for a	in [ xml_table.get_column('inclination')[i],
		xml_table.get_column('spin1x')[i], xml_table.get_column('spin1y')[i], xml_table.get_column('spin1z')[i],
		xml_table.get_column('spin2x')[i], xml_table.get_column('spin2y')[i], xml_table.get_column('spin2z')[i],
		mass1_samples[i], mass2_samples[i], ref_freq, xml_table.get_column('coa_phase')[i] ] ]

	theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = lalsim.SimInspiralTransformPrecessingWvf2PE(*param_array )
	
	phi_jl_array.append(phi_jl)
	tilt_1_array.append(tilt_1)
	tilt_2_array.append(tilt_2)
	phi_12_array.append(phi_12)
	a_1_array.append(a_1)
	a_2_array.append(a_2)


print ("min max a_1_array", min(a_1_array), max(a_1_array))
print ("min max a_2_array", min(a_2_array), max(a_2_array))



print ('starting chirp length calculation')

chirp_time_array = tau0_from_mass1_mass2(mass1_samples, mass2_samples,	np.ones_like( mass1_samples )*options.flow)


print ('chirp_time_array len',len(chirp_time_array))


indices_4s = []
indices_8s = []
indices_16s = []
indices_32s = []
indices_64s = []
indices_128s = []



for i, tau0 in enumerate(chirp_time_array):
	if tau0 <= 4:
		indices_4s.append(i)
	if tau0 > 4 and tau0 <= 8:
		indices_8s.append(i)
	if tau0 > 8 and tau0 <= 16:
		indices_16s.append(i)
	if tau0 > 16 and tau0 <= 32:
		indices_32s.append(i)
	if tau0 > 32 and tau0 <= 64:
		indices_64s.append(i)
	if tau0 >64:
		indices_128s.append(i)
	#if tau0 >= 64 and tau0 <= 128 :
	#	 indices_128s.append(i)


if len (indices_4s) !=0:
	
	indices_4s = np.asarray(indices_4s, dtype=int)

	chirp_mass_samples, q_samples, m1_samples, m2_samples = mass_params( mass1_samples[indices_4s], mass2_samples[indices_4s] )


	luminosity_distance_samples = distance_samples[indices_4s] #Mpc conversion
	
	H1 = np.ones_like( m1_samples ); H2 = np.ones_like( m2_samples )

	Heff5_samples, Heff8_samples = get_h_params( m1_samples, m2_samples, xml_table.get_column('spin1z')[indices_4s], 
					xml_table.get_column('spin2z')[indices_4s], H1, H2 )

	bbh_param_dict = {"H_eff5": Heff5_samples, "H_eff8": Heff8_samples	}
	
	lambda1_samples, lambda2_samples = get_lambda1_lambda2_from_mass1_mass2 ( m1_samples, m2_samples, eos_name = options.eos )

	print ("min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))

	bns_param_dict = {"lambda_1":lambda1_samples, "lambda_2":lambda2_samples }
	
	inj_dict = { "mass_1":m1_samples, "mass_2":m2_samples, "chirp_mass": chirp_mass_samples, "mass_ratio": q_samples,
				"luminosity_distance": luminosity_distance_samples, "dec": dec_samples[indices_4s], "ra": ra_samples[indices_4s],
				"psi":xml_table.get_column('polarization')[indices_4s], "phase": xml_table.get_column('coa_phase')[indices_4s],
				"theta_jn": xml_table.get_column('inclination')[indices_4s], "a_1": np.asarray(a_1_array)[indices_4s], "a_2": np.asarray(a_2_array)[indices_4s],
				"tilt_1": np.asarray(tilt_1_array)[indices_4s], "tilt_2": np.asarray(tilt_2_array)[indices_4s], "phi_12": np.asarray(phi_12_array)[indices_4s],
				"phi_jl":np.asarray(phi_jl_array)[indices_4s], "geocent_time": np.zeros_like(a_2_array)[indices_4s],
				"chi_1":chi1_array[indices_4s], "chi_2":chi2_array[indices_4s] }

	bbh_param_dict.update(inj_dict)
	bns_param_dict.update(inj_dict)

	df_injections_bbh = pd.DataFrame.from_dict(bbh_param_dict, orient='columns')
	df_injections_bbh.to_csv( os.path.join( inj_dir, "injection_bbh_4s.dat"), sep=' ', index=False)

	df_injections_bns = pd.DataFrame.from_dict(bns_param_dict, orient='columns')
	df_injections_bns.to_csv( os.path.join( inj_dir, "injection_bns_eos_{}_4s.dat".format(options.eos)), sep=' ', index=False)
	np.savetxt( os.path.join( inj_dir, "gps_file_4s.txt"), geocent_end_time_samples[indices_4s])

	print ("[4s] min max chirp_mass_samples", min(chirp_mass_samples), max(chirp_mass_samples))
	print ("[4s] min max q_samples", min(q_samples), max(q_samples))
	print ("[4s] min max m1_samples", min(m1_samples), max(m1_samples))
	print ("[4s] min max m2_samples", min(m2_samples), max(m2_samples))
	print ("[4s] min max chi_1 samples", min(chi1_array[indices_4s]), max(chi1_array[indices_4s]))
	print ("[4s] min max chi_2 samples", min(chi2_array[indices_4s]), max(chi2_array[indices_4s]))
	print ("[4s] min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("[4s] min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))



if len (indices_8s) !=0:
	
	indices_8s = np.asarray(indices_8s, dtype=int)

	chirp_mass_samples, q_samples, m1_samples, m2_samples = mass_params( mass1_samples[indices_8s], mass2_samples[indices_8s] )

	luminosity_distance_samples = distance_samples[indices_8s] #Mpc conversion
	
	H1 = np.ones_like( m1_samples ); H2 = np.ones_like( m2_samples )

	Heff5_samples, Heff8_samples = get_h_params( m1_samples, m2_samples, xml_table.get_column('spin1z')[indices_8s], 
					xml_table.get_column('spin2z')[indices_8s], H1, H2 )

	bbh_param_dict = {"H_eff5": Heff5_samples, "H_eff8": Heff8_samples	}
	
	lambda1_samples, lambda2_samples = get_lambda1_lambda2_from_mass1_mass2 ( m1_samples, m2_samples, eos_name = options.eos )


	bns_param_dict = {"lambda_1":lambda1_samples, "lambda_2":lambda2_samples }
	
	inj_dict = { "mass_1":m1_samples, "mass_2":m2_samples, "chirp_mass": chirp_mass_samples, "mass_ratio": q_samples,
				"luminosity_distance": luminosity_distance_samples, "dec": dec_samples[indices_8s], "ra": ra_samples[indices_8s],
				"psi":xml_table.get_column('polarization')[indices_8s], "phase": xml_table.get_column('coa_phase')[indices_8s],
				"theta_jn": xml_table.get_column('inclination')[indices_8s], "a_1": np.asarray(a_1_array)[indices_8s], "a_2": np.asarray(a_2_array)[indices_8s],
				"tilt_1": np.asarray(tilt_1_array)[indices_8s], "tilt_2": np.asarray(tilt_2_array)[indices_8s], "phi_12": np.asarray(phi_12_array)[indices_8s],
				"phi_jl":np.asarray(phi_jl_array)[indices_8s], "geocent_time": np.zeros_like(a_2_array)[indices_8s],
				"chi_1":chi1_array[indices_8s], "chi_2":chi2_array[indices_8s] }

	bbh_param_dict.update(inj_dict)
	bns_param_dict.update(inj_dict)

	df_injections_bbh = pd.DataFrame.from_dict(bbh_param_dict, orient='columns')
	df_injections_bbh.to_csv( os.path.join( inj_dir, "injection_bbh_8s.dat"), sep=' ', index=False)

	df_injections_bns = pd.DataFrame.from_dict(bns_param_dict, orient='columns')
	df_injections_bns.to_csv( os.path.join( inj_dir, "injection_bns_eos_{}_8s.dat".format(options.eos)), sep=' ', index=False)

	np.savetxt( os.path.join( inj_dir, "gps_file_8s.txt"), geocent_end_time_samples[indices_8s])

	print ("[8s] min max chirp_mass_samples", min(chirp_mass_samples), max(chirp_mass_samples))
	print ("[8s] min max q_samples", min(q_samples), max(q_samples))
	print ("[8s] min max m1_samples", min(m1_samples), max(m1_samples))
	print ("[8s] min max m2_samples", min(m2_samples), max(m2_samples))
	print ("[8s] min max chi_1 samples", min( chi1_array[indices_8s] ), max( chi1_array[indices_8s] ) )
	print ("[8s] min max chi_2 samples", min( chi2_array[indices_8s] ), max( chi2_array[indices_8s] ) )
	print ("[8s] min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("[8s] min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))



if len (indices_16s) !=0:
	
	indices_16s = np.asarray(indices_16s, dtype=int)

	chirp_mass_samples, q_samples, m1_samples, m2_samples = mass_params( mass1_samples[indices_16s], mass2_samples[indices_16s] )


	luminosity_distance_samples = distance_samples[indices_16s]#Mpc conversion
	
	H1 = np.ones_like( m1_samples ); H2 = np.ones_like( m2_samples )


	Heff5_samples, Heff8_samples = get_h_params( m1_samples, m2_samples, xml_table.get_column('spin1z')[indices_16s], 
					xml_table.get_column('spin2z')[indices_16s], H1, H2 )

	bbh_param_dict = {"H_eff5": Heff5_samples, "H_eff8": Heff8_samples	}
	
	lambda1_samples, lambda2_samples = get_lambda1_lambda2_from_mass1_mass2 ( m1_samples, m2_samples, eos_name = options.eos )


	bns_param_dict = {"lambda_1":lambda1_samples, "lambda_2":lambda2_samples }
	
	inj_dict = { "mass_1":m1_samples, "mass_2":m2_samples, "chirp_mass": chirp_mass_samples, "mass_ratio": q_samples,
				"luminosity_distance": luminosity_distance_samples, "dec": dec_samples[indices_16s], "ra": ra_samples[indices_16s],
				"psi":xml_table.get_column('polarization')[indices_16s], "phase": xml_table.get_column('coa_phase')[indices_16s],
				"theta_jn": xml_table.get_column('inclination')[indices_16s], "a_1": np.asarray(a_1_array)[indices_16s], "a_2": np.asarray(a_2_array)[indices_16s],
				"tilt_1": np.asarray(tilt_1_array)[indices_16s], "tilt_2": np.asarray(tilt_2_array)[indices_16s], "phi_12": np.asarray(phi_12_array)[indices_16s],
				"phi_jl":np.asarray(phi_jl_array)[indices_16s], "geocent_time": np.zeros_like(a_2_array)[indices_16s],
				"chi_1":chi1_array[indices_16s], "chi_2":chi2_array[indices_16s] }

	bbh_param_dict.update(inj_dict)
	bns_param_dict.update(inj_dict)

	df_injections_bbh = pd.DataFrame.from_dict(bbh_param_dict, orient='columns')
	df_injections_bbh.to_csv( os.path.join( inj_dir, "injection_bbh_16s.dat"), sep=' ', index=False)

	df_injections_bns = pd.DataFrame.from_dict(bns_param_dict, orient='columns')
	df_injections_bns.to_csv( os.path.join( inj_dir, "injection_bns_eos_{}_16s.dat".format(options.eos)), sep=' ', index=False)

	np.savetxt( os.path.join( inj_dir, "gps_file_16s.txt"), geocent_end_time_samples[indices_16s])

	print ("[16s] min max chirp_mass_samples", min(chirp_mass_samples), max(chirp_mass_samples))
	print ("[16s] min max q_samples", min(q_samples), max(q_samples))
	print ("[16s] min max m1_samples", min(m1_samples), max(m1_samples))
	print ("[16s] min max m2_samples", min(m2_samples), max(m2_samples))
	print ("[16s] min max chi_1 samples", min( chi1_array[indices_16s] ), max( chi1_array[indices_16s] ) )
	print ("[16s] min max chi_2 samples", min( chi2_array[indices_16s] ), max( chi2_array[indices_16s] ) )
	print ("[16s] min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("[16s] min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))


if len (indices_32s) !=0:
	
	indices_32s = np.asarray(indices_32s, dtype=int)

	chirp_mass_samples, q_samples, m1_samples, m2_samples = mass_params( mass1_samples[indices_32s], mass2_samples[indices_32s] )


	luminosity_distance_samples = distance_samples[indices_32s] #Mpc conversion
	
	H1 = np.ones_like( m1_samples ); H2 = np.ones_like( m2_samples )

	Heff5_samples, Heff8_samples = get_h_params( m1_samples, m2_samples, xml_table.get_column('spin1z')[indices_32s], 
					xml_table.get_column('spin2z')[indices_32s], H1, H2 )

	lambda1_samples, lambda2_samples = get_lambda1_lambda2_from_mass1_mass2 ( m1_samples, m2_samples, eos_name = options.eos )
	
        bbh_param_dict = {"H_eff5": Heff5_samples, "H_eff8": Heff8_samples,
                            "lambda_1":np.zeros_like( lambda1_samples ), "lambda_2":np.zeros_like( lambda2_samples) }


	bns_param_dict = {"lambda_1":lambda1_samples, "lambda_2":lambda2_samples,
                                "H_eff5": np.zeros_like(Heff5_samples), "H_eff8":np.zeros_like(Heff8_samples)}
	
	inj_dict = { "mass_1":m1_samples, "mass_2":m2_samples, "chirp_mass": chirp_mass_samples, "mass_ratio": q_samples,
				"luminosity_distance": luminosity_distance_samples, "dec": dec_samples[indices_32s], "ra": ra_samples[indices_32s],
				"psi":xml_table.get_column('polarization')[indices_32s], "phase": xml_table.get_column('coa_phase')[indices_32s],
				"theta_jn": xml_table.get_column('inclination')[indices_32s], "a_1": np.asarray(a_1_array)[indices_32s], "a_2": np.asarray(a_2_array)[indices_32s],
				"tilt_1": np.asarray(tilt_1_array)[indices_32s], "tilt_2": np.asarray(tilt_2_array)[indices_32s], "phi_12": np.asarray(phi_12_array)[indices_32s],
				"phi_jl":np.asarray(phi_jl_array)[indices_32s], "geocent_time": np.zeros_like(a_2_array)[indices_32s],
				"chi_1":chi1_array[indices_32s], "chi_2":chi2_array[indices_32s] }

	bbh_param_dict.update(inj_dict)
	bns_param_dict.update(inj_dict)

	df_injections_bbh = pd.DataFrame.from_dict(bbh_param_dict, orient='columns')
	df_injections_bbh.to_csv( os.path.join( inj_dir, "injection_bbh_32s.dat"), sep=' ', index=False)

	df_injections_bns = pd.DataFrame.from_dict(bns_param_dict, orient='columns')
	df_injections_bns.to_csv( os.path.join( inj_dir, "injection_bns_eos_{}_32s.dat".format(options.eos)), sep=' ', index=False)

	np.savetxt( os.path.join( inj_dir, "gps_file_32s.txt"), geocent_end_time_samples[indices_32s])


	print ("[32s] min max chirp_mass_samples", min(chirp_mass_samples), max(chirp_mass_samples))
	print ("[32s] min max q_samples", min(q_samples), max(q_samples))
	print ("[32s] min max m1_samples", min(m1_samples), max(m1_samples))
	print ("[32s] min max m2_samples", min(m2_samples), max(m2_samples))
	print ("[32s] min max chi_1 samples", min( chi1_array[indices_32s] ), max( chi1_array[indices_32s] ) )
	print ("[32s] min max chi_2 samples", min( chi2_array[indices_32s] ), max( chi2_array[indices_32s] ) )
	print ("[32s] min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("[32s] min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))



if len (indices_64s) !=0:
	
	indices_64s = np.asarray(indices_64s, dtype=int)

	chirp_mass_samples, q_samples, m1_samples, m2_samples = mass_params( mass1_samples[indices_64s], mass2_samples[indices_64s] )


	luminosity_distance_samples = distance_samples[indices_64s] #Mpc conversion
	
	H1 = np.ones_like( m1_samples ); H2 = np.ones_like( m2_samples )
	
	Heff5_samples, Heff8_samples = get_h_params( m1_samples, m2_samples, xml_table.get_column('spin1z')[indices_64s], 
					xml_table.get_column('spin2z')[indices_64s], H1, H2 )
	
	lambda1_samples, lambda2_samples = get_lambda1_lambda2_from_mass1_mass2 ( m1_samples, m2_samples, eos_name = options.eos )
	
        bbh_param_dict = {"H_eff5": Heff5_samples, "H_eff8": Heff8_samples,
                    "lambda_1":np.zeros_like( lambda1_samples ), "lambda_2":np.zeros_like( lambda2_samples)}


	bns_param_dict = {"lambda_1":lambda1_samples, "lambda_2":lambda2_samples,  
                    "H_eff5": np.zeros_like(Heff5_samples), "H_eff8":np.zeros_like(Heff8_samples) }
	
	inj_dict = { "mass_1":m1_samples, "mass_2":m2_samples, "chirp_mass": chirp_mass_samples, "mass_ratio": q_samples,
				"luminosity_distance": luminosity_distance_samples, "dec": dec_samples[indices_64s], "ra": ra_samples[indices_64s],
				"psi":xml_table.get_column('polarization')[indices_64s], "phase": xml_table.get_column('coa_phase')[indices_64s],
				"theta_jn": xml_table.get_column('inclination')[indices_64s], "a_1": np.asarray(a_1_array)[indices_64s], "a_2": np.asarray(a_2_array)[indices_64s],
				"tilt_1": np.asarray(tilt_1_array)[indices_64s], "tilt_2": np.asarray(tilt_2_array)[indices_64s], "phi_12": np.asarray(phi_12_array)[indices_64s],
				"phi_jl":np.asarray(phi_jl_array)[indices_64s], "geocent_time": np.zeros_like(a_2_array)[indices_64s],
				"chi_1":chi1_array[indices_64s], "chi_2":chi2_array[indices_64s] }

	bbh_param_dict.update(inj_dict)
	bns_param_dict.update(inj_dict)

	df_injections_bbh = pd.DataFrame.from_dict(bbh_param_dict, orient='columns')
	df_injections_bbh.to_csv( os.path.join( inj_dir, "injection_bbh_64s.dat"), sep=' ', index=False)

	df_injections_bns = pd.DataFrame.from_dict(bns_param_dict, orient='columns')
	df_injections_bns.to_csv( os.path.join( inj_dir, "injection_bns_eos_{}_64s.dat".format(options.eos)), sep=' ', index=False)

	np.savetxt( os.path.join( inj_dir, "gps_file_64s.txt"), geocent_end_time_samples[indices_64s])

	print ("[64s] min max chirp_mass_samples", min(chirp_mass_samples), max(chirp_mass_samples))
	print ("[64s] min max q_samples", min(q_samples), max(q_samples))
	print ("[64s] min max m1_samples", min(m1_samples), max(m1_samples))
	print ("[64s] min max m2_samples", min(m2_samples), max(m2_samples))
	print ("[64s] min max chi_1 samples", min( chi1_array[indices_64s] ), max( chi1_array[indices_64s] ) )
	print ("[64s] min max chi_2 samples", min( chi2_array[indices_64s] ), max( chi2_array[indices_64s] ) )
	print ("[64s] min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("[64s] min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))


if len (indices_128s) !=0:
	
	indices_128s = np.asarray(indices_128s, dtype=int)

	chirp_mass_samples, q_samples, m1_samples, m2_samples = mass_params( mass1_samples[indices_128s], mass2_samples[indices_128s] )


	luminosity_distance_samples = distance_samples[indices_128s] #Mpc conversion
	
	H1 = np.ones_like( m1_samples ); H2 = np.ones_like( m2_samples )
	
	Heff5_samples, Heff8_samples = get_h_params( m1_samples, m2_samples, xml_table.get_column('spin1z')[indices_128s], 
					xml_table.get_column('spin2z')[indices_128s], H1, H2 )
	
	lambda1_samples, lambda2_samples = get_lambda1_lambda2_from_mass1_mass2 ( m1_samples, m2_samples, eos_name = options.eos )

	print ("min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))

	bbh_param_dict = {"H_eff5": Heff5_samples, "H_eff8": Heff8_samples,
                "lambda_1":np.zeros_like( lambda1_samples ), "lambda_2":np.zeros_like( lambda2_samples)}

	bns_param_dict = {"lambda_1":lambda1_samples, "lambda_2":lambda2_samples, 
                "H_eff5": np.zeros_like(Heff5_samples), "H_eff8":np.zeros_like(Heff8_samples) }
	
	inj_dict = { "mass_1":m1_samples, "mass_2":m2_samples, "chirp_mass": chirp_mass_samples, "mass_ratio": q_samples,
				"luminosity_distance": luminosity_distance_samples, "dec": dec_samples[indices_128s], "ra": ra_samples[indices_128s],
				"psi":xml_table.get_column('polarization')[indices_128s], "phase": xml_table.get_column('coa_phase')[indices_128s],
				"theta_jn": xml_table.get_column('inclination')[indices_128s], "a_1": np.asarray(a_1_array)[indices_128s], "a_2": np.asarray(a_2_array)[indices_128s],
				"tilt_1": np.asarray(tilt_1_array)[indices_128s], "tilt_2": np.asarray(tilt_2_array)[indices_128s], "phi_12": np.asarray(phi_12_array)[indices_128s],
				"phi_jl":np.asarray(phi_jl_array)[indices_128s], "geocent_time": np.zeros_like(a_2_array)[indices_128s],
				"chi_1":chi1_array[indices_128s], "chi_2":chi2_array[indices_128s] }

	bbh_param_dict.update(inj_dict)
	bns_param_dict.update(inj_dict)

	df_injections_bbh = pd.DataFrame.from_dict(bbh_param_dict, orient='columns')
	df_injections_bbh.to_csv( os.path.join( inj_dir, "injection_bbh_128s.dat"), sep=' ', index=False)

	df_injections_bns = pd.DataFrame.from_dict(bns_param_dict, orient='columns')
	df_injections_bns.to_csv( os.path.join( inj_dir, "injection_bns_eos_{}_128s.dat".format(options.eos)), sep=' ', index=False)

	np.savetxt( os.path.join( inj_dir, "gps_file_128s.txt"), geocent_end_time_samples[indices_128s])


	print ("[128s] min max chirp_mass_samples", min(chirp_mass_samples), max(chirp_mass_samples))
	print ("[128s] min max q_samples", min(q_samples), max(q_samples))
	print ("[128s] min max m1_samples", min(m1_samples), max(m1_samples))
	print ("[128s] min max m2_samples", min(m2_samples), max(m2_samples))
	print ("[128s] min max chi_1 samples", min( chi1_array[indices_128s] ), max( chi1_array[indices_128s] ) )
	print ("[128s] min max chi_2 samples", min( chi2_array[indices_128s] ), max( chi2_array[indices_128s] ) )
	print ("[128s] min max lambda1_samples", min(lambda1_samples), max(lambda1_samples))
	print ("[128s] min max lambda2_samples", min(lambda2_samples), max(lambda2_samples))


