import pandas as pd

# create dataframes for lh/rh cortical gm thickness/area and ms_studies
rh_thickness = pd.read_csv('hc/rh_thickness_hc.csv')
rh_thickness = rh_thickness.rename(columns={'rh.aparc.a2009s.thickness': 'id'})
lh_thickness = pd.read_csv('hc/lh_thickness_hc.csv')
lh_thickness = lh_thickness.rename(columns={'lh.aparc.a2009s.thickness': 'id'})
rh_area = pd.read_csv('hc/rh_area_hc.csv')
rh_area = rh_area.rename(columns={'rh.aparc.a2009s.area': 'id'})
lh_area = pd.read_csv('hc/lh_area_hc.csv')
lh_area = lh_area.rename(columns={'lh.aparc.a2009s.area': 'id'})


# create dataframes for subcortical structures (rh/lh hippocampus, amygdala, thalamus, hypothalamus)
lh_amy = pd.read_csv('hc/lh_amy_hc.csv')
lh_amy = lh_amy.add_prefix('lh_')
lh_amy = lh_amy.rename(columns={'lh_Measure:volume': 'id'})
rh_amy = pd.read_csv('hc/rh_amy_hc.csv')
rh_amy = rh_amy.add_prefix('rh_')
rh_amy = rh_amy.rename(columns={'rh_Measure:volume': 'id'})
lh_hipp = pd.read_csv('hc/lh_hipp_hc.csv')
lh_hipp = lh_hipp.add_prefix('lh_')
lh_hipp = lh_hipp.rename(columns={'lh_Measure:volume': 'id'})
rh_hipp = pd.read_csv('hc/rh_hipp_hc.csv')
rh_hipp = rh_hipp.add_prefix('rh_')
rh_hipp = rh_hipp.rename(columns={'rh_Measure:volume': 'id'})
lh_thal = pd.read_csv('hc/lh_thal_hc.csv')
lh_thal = lh_thal.add_prefix('lh_')
lh_thal = lh_thal.rename(columns={'lh_Measure:volume': 'id'})
rh_thal = pd.read_csv('hc/rh_thal_hc.csv')
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

# merge matching columns with different names
for column in gm_data.columns:
    # replace '_and_' with '&'
    gm_data = gm_data.rename(columns={column: column.replace('_and_', '&')})
gm_data = gm_data.groupby(level=0, axis=1).sum()
gm_data = gm_data.dropna(axis=1)

# create and save final dataframe
hc_dataframe = pd.merge(gm_data, subcortical_data, on='id', how='outer')
print(hc_dataframe.shape)
hc_dataframe = hc_dataframe.drop(columns=["rh_WhiteSurfArea_area", "lh_WhiteSurfArea_area", "lh_fimbria", "rh_fimbria"])
hc_dataframe.to_csv('hc/hc_dataframe.csv', index=False)
