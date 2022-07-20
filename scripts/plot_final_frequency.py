import matplotlib
matplotlib.use("agg")
import numpy as np
import pandas as pd
import pylab as plt
import pycbc
import pycbc.conversions
import pycbc.pnutils
import json

pe_samples_file = '/data/gravwav/ksphukon/Work/TidalHeating/TidalHeatingGWTC/TidalHeatingPE/GW170814/pe/outdir_C/result/dynesty_HeatedTaylorF2_GW170814_data0_1186741861-53_analysis_H1L1V1_dynesty_par0_check_result.json'

with open(pe_samples_file) as f:
	pe_samples = json.load(f)

samples_pe = np.asarray(pe_samples['samples']['content'])

samples_data_frame = pd.DataFrame( samples_pe, columns=pe_samples['search_parameter_keys'])

print (samples_data_frame.head())

chirp_mass_array =  samples_data_frame['chirp_mass']
mass_ratio_array = samples_data_frame['mass_ratio']
spin1z_array = samples_data_frame['chi_1'] 
spin2z_array = samples_data_frame['chi_2'] 


array_size = len(spin1z_array)

mass1_array = pycbc.conversions.mass1_from_mchirp_q(chirp_mass_array, mass_ratio_array)
mass2_array = pycbc.conversions.mass2_from_mchirp_q(chirp_mass_array, mass_ratio_array)

f_final_dict = dict()

named_frequency_cutoffs = ['SchwarzISCO', 'MECO', 'IMRPhenomDPeak', 'SEOBNRv1Peak', 'SEOBNRv4Peak', 'BKLISCO', 'HybridMECO']

for name_key in named_frequency_cutoffs:
	f_list = []	
	for i in range(array_size): 
		f = pycbc.pnutils.frequency_cutoff_from_name(name_key, mass1_array[i],
						mass2_array[i], 
						spin1z_array[i],
						spin2z_array[i])
		f_list.append(f)

	f_final_dict[name_key] = f_list




plt.figure()
for name_key in named_frequency_cutoffs:
	plt.scatter(  chirp_mass_array, f_final_dict[name_key], label = name_key  )
plt.ylabel('final filter frequeny')
plt.xlabel('Chirp mass')
plt.legend(loc='best')
plt.savefig("/data/gravwav/ksphukon/Work/TidalHeating/git_repo/tidal_heating/plots/final_freq_filters.png")



