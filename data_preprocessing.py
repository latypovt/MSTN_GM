import numpy as np
import pandas as pd

# create dataframes
rh_thickness = pd.read_csv('stats/rh_thickness.csv')
rh_thickness = rh_thickness.rename(columns={'rh.aparc.a2009s.thickness': 'id'})
lh_thickness = pd.read_csv('stats/lh_thickness.csv')
lh_thickness = lh_thickness.rename(columns={'lh.aparc.a2009s.thickness': 'id'})
rh_area = pd.read_csv('stats/rh_area.csv')
rh_area = rh_area.rename(columns={'rh.aparc.a2009s.area': 'id'})
lh_area = pd.read_csv('stats/lh_area.csv')
lh_area = lh_area.rename(columns={'lh.aparc.a2009s.area': 'id'})
ms_studies = pd.read_csv('stats/MS_studies.csv')

# drop unnecessary columns
rh_area = rh_area.drop(columns=['BrainSegVolNotVent', 'eTIV'])
lh_area = lh_area.drop(columns=['BrainSegVolNotVent', 'eTIV'])
lh_thickness = lh_thickness.drop(columns=['BrainSegVolNotVent', 'eTIV'])
rh_thickness = rh_thickness.drop(columns=['BrainSegVolNotVent'])

# merge dataframes
thickness = pd.merge(rh_thickness, lh_thickness, on='id')
area = pd.merge(rh_area, lh_area, on='id')
gm_data = pd.merge(thickness, area, on='id')

# drop unnecessary columns
gm_features = gm_data.columns
gm_features = gm_features.drop('id')
gm_features = gm_features.drop('eTIV')

# normalize data
for feature in gm_features:
    gm_data[feature] = (gm_data[feature])*1000 / gm_data['eTIV']

# create and save final dataframe
ml_dataframe = pd.merge(ms_studies, gm_data, on='id')
ml_dataframe.to_csv('stats/ml_dataframe.csv', index=False)