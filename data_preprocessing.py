import pandas as pd

# create dataframes for lh/rh cortical gm thickness/area and ms_studies
rh_thickness = pd.read_csv('stats/rh_thickness.csv')
rh_thickness = rh_thickness.rename(columns={'rh.aparc.a2009s.thickness': 'id'})
lh_thickness = pd.read_csv('stats/lh_thickness.csv')
lh_thickness = lh_thickness.rename(columns={'lh.aparc.a2009s.thickness': 'id'})
rh_area = pd.read_csv('stats/rh_area.csv')
rh_area = rh_area.rename(columns={'rh.aparc.a2009s.area': 'id'})
lh_area = pd.read_csv('stats/lh_area.csv')
lh_area = lh_area.rename(columns={'lh.aparc.a2009s.area': 'id'})
ms_studies = pd.read_csv('stats/MS_studies.csv')

# create dataframes for subcortical structures (rh/lh hippocampus, amygdala, thalamus, hypothalamus)
lh_amy = pd.read_csv('stats/lh_amy.csv')
lh_amy = lh_amy.add_prefix('lh_')
lh_amy = lh_amy.rename(columns={'lh_Measure:volume': 'id'})
rh_amy = pd.read_csv('stats/rh_amy.csv')
rh_amy = rh_amy.add_prefix('rh_')
rh_amy = rh_amy.rename(columns={'rh_Measure:volume': 'id'})
lh_hipp = pd.read_csv('stats/lh_hipp.csv')
lh_hipp = lh_hipp.add_prefix('lh_')
lh_hipp = lh_hipp.rename(columns={'lh_Measure:volume': 'id'})
rh_hipp = pd.read_csv('stats/rh_hipp.csv')
rh_hipp = rh_hipp.add_prefix('rh_')
rh_hipp = rh_hipp.rename(columns={'rh_Measure:volume': 'id'})
lh_thal = pd.read_csv('stats/lh_thal.csv')
lh_thal = lh_thal.add_prefix('lh_')
lh_thal = lh_thal.rename(columns={'lh_Measure:volume': 'id'})
rh_thal = pd.read_csv('stats/rh_thal.csv')
rh_thal = rh_thal.add_prefix('rh_')
rh_thal = rh_thal.rename(columns={'rh_Measure:volume': 'id'})

# drop unnecessary columns
rh_area = rh_area.drop(columns=['BrainSegVolNotVent', 'eTIV'])
lh_area = lh_area.drop(columns=['BrainSegVolNotVent', 'eTIV'])
lh_thickness = lh_thickness.drop(columns=['BrainSegVolNotVent', 'eTIV'])
rh_thickness = rh_thickness.drop(columns=['BrainSegVolNotVent'])

# merge dataframes
thickness = pd.merge(rh_thickness, lh_thickness, on='id')
area = pd.merge(rh_area, lh_area, on='id')
gm_data = pd.merge(thickness, area, on='id')

amygdala = pd.merge(rh_amy, lh_amy, on='id')
hippocampus = pd.merge(rh_hipp, lh_hipp, on='id')
thalamus = pd.merge(rh_thal, lh_thal, on='id')
subcortical_data = pd.merge(amygdala, hippocampus, on='id')
subcortical_data = pd.merge(subcortical_data, thalamus, on='id')

# create dictionary for eTIV of each id
eTIV_dict = {}
for index, row in gm_data.iterrows():
    eTIV_dict[row['id']] = row['eTIV']

# drop unnecessary columns
gm_features = gm_data.columns
gm_features = gm_features.drop('id')
gm_features = gm_features.drop('eTIV')

subcortical_features = subcortical_data.columns
subcortical_features = subcortical_features.drop('id')

# normalize data
for feature in gm_features:
    gm_data[feature] = (gm_data[feature])*1000 / gm_data['eTIV']

for feature in subcortical_features:
    subcortical_data[feature] = (subcortical_data[feature])*1000 / subcortical_data['id'].map(eTIV_dict)

# create and save final dataframe
ml_dataframe = pd.merge(ms_studies, gm_data, on='id')
print(ml_dataframe.shape)
ml_dataframe = pd.merge(ml_dataframe, subcortical_data, on='id')
print(ml_dataframe.shape)
ml_dataframe.to_csv('stats/ml_dataframe.csv', index=False)