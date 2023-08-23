# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import matplotlib.font_manager


def main(): 
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--k_features", type=str, default="parsimonious")
    parser.add_argument("--path_to_data", type=str, default="stats/ml_dataframe.csv")
    parser.add_argument("--kernel", type=str, default="linear")
    parser.add_argument("--C", type=float, default=0.1)
    args = parser.parse_args()

    # create dataframes
    ml_dataframe = pd.read_csv(args.path_to_data)
    ml_dataframe = ml_dataframe.drop(columns=['eTIV'])
    ml_dataframe['sex'] = [1 if sex=="f" else 0 for sex in ml_dataframe["sex"]]
    ml_dataframe['duration_categories'] = ['<5' if duration <5 else '5-10' if duration < 10 else '10-15' if duration <15 else '15-20' if duration<20 else '>20' for duration in ml_dataframe['duration_of_ms']]
    ml_dataframe['edss_categories'] = ['0-1.5' if edss <1.5 else '1.5-3' if edss < 3 else '3-4.5' if edss <4.5 else '4.5-6' if edss<6 else '6-7.5' if edss<=7.5 else 'MS-TN' for edss in ml_dataframe['edss']]

    # scanner column
    scanner = []
    for id in ml_dataframe['id']:
       if 'ms-tn-' in id:
          scanner.append('Siemens Vida')
       elif 'ms_' in id:
          scanner.append('Siemens Trio')
       else:
          scanner.append('GE Signa')
    ml_dataframe['scanner'] = scanner
       

    # set condition
    condition = list(ml_dataframe["diagnosis"])
    condition = np.array(condition)

    # drop unnecessary columns
    morphological_data = ml_dataframe.drop(columns=['id', 'age', 'sex', 'diagnosis', 'duration_of_ms', 'duration_categories', 'duration_of_pain', 'side_of_pain','edss', 'edss_categories', 'scanner'])
    morphological_data = morphological_data[np.random.default_rng(seed=42).permutation(morphological_data.columns.values)]

    morphological_data = morphological_data.dropna(axis=1, how='any')
    # configure tsne model
    print("initializing...")
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_data = tsne.fit_transform(morphological_data)

    # create dataframe
    tsne_data = np.vstack((tsne_data.T, ml_dataframe["diagnosis"])).T    
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "diagnosis"))
    tsne_df["edss_categories"] = ml_dataframe["edss_categories"]
    tsne_df["age"] = ml_dataframe["age"]
    tsne_df["sex"] = ml_dataframe["sex"]
    tsne_df["duration_categories"] = ml_dataframe["duration_categories"]
    tsne_df["scanner"] = ml_dataframe["scanner"]

    matplotlib.font_manager.findfont('humor sans')

    # plot
    with plt.xkcd():
      plt.figure(figsize=(10,10))
      sns.scatterplot(tsne_df, x="Dim_1", y="Dim_2", hue='diagnosis', sizes = 5, alpha=0.8)
      for i, subject in enumerate (ml_dataframe["id"]):
          plt.annotate(subject, (tsne_data[i,0]+0.08, tsne_data[i,1]+0.08), fontsize=6)
      plt.savefig('out/tsne.png', dpi=300)

    with plt.xkcd():
      fig, ax = plt.subplots(2, 2, figsize=(15, 15))

      sns.scatterplot(ax=ax[1, 0], data=tsne_df, x="Dim_1", y="Dim_2", hue='duration_categories', hue_order = ['<5', '5-10', '10-15', '10-15', '15-20', '>20'], sizes=5, alpha=0.8)
      ax[1, 0].legend(loc='upper right', title='Duration of MS', fontsize='xx-small')


      sns.scatterplot(ax=ax[0, 0], data=tsne_df, x="Dim_1", y="Dim_2", hue='diagnosis', sizes=5, alpha=0.8)
      ax[0, 0].legend(loc='upper right', title='Diagnosis', fontsize='xx-small')
      sns.scatterplot(ax=ax[0, 1], data=tsne_df, x="Dim_1", y="Dim_2", hue='scanner', sizes=5, alpha=0.8)
      ax[0, 1].legend(loc='upper right', title='Scanner', fontsize='xx-small')
      sns.scatterplot(ax=ax[1, 1], data=tsne_df, x="Dim_1", y="Dim_2", hue='age', sizes=5, alpha=0.8)
      ax[1, 1].legend(loc='upper right', title='Age', fontsize='xx-small')
      plt.savefig('out/tsne_subplots.png', dpi=300)


 



if __name__ == "__main__":
  main()