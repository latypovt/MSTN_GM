# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse



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
    ml_dataframe['diagnosis'] = [1 if dx =="MS-TN" else 0 for dx in ml_dataframe['diagnosis']]
    ml_dataframe['sex'] = [1 if sex=="f" else 0 for sex in ml_dataframe["sex"]]
    ml_dataframe['duration_categories'] = ['<5' if duration <5 else '5-10' if duration < 10 else '10-15' if duration <15 else '15-20' if duration<20 else '>20' for duration in ml_dataframe['duration_of_ms']]

    # set condition
    #condition = list(ml_dataframe["diagnosis"])
    #condition = np.array(condition)

    # drop unnecessary columns
    morphological_data = ml_dataframe.drop(columns=['id', 'age', 'sex', 'diagnosis', 'duration_of_ms', 'duration_categories', 'duration_of_pain', 'side_of_pain','edss'])
    morphological_data = morphological_data[np.random.default_rng(seed=42).permutation(morphological_data.columns.values)]

    # configure tsne model
    print("initializing...")
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_data = tsne.fit_transform(morphological_data)

    # create dataframe
    tsne_data = np.vstack((tsne_data.T, ml_dataframe["edss"])).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "diagnosis"))

    # plot
    with plt.xkcd():
      plt.figure(figsize=(10,10))
      sns.scatterplot(tsne_df, x="Dim_1", y="Dim_2", hue='diagnosis', alpha=0.8)
      for i, subject in enumerate (ml_dataframe["id"]):
          plt.annotate(subject, (tsne_data[i,0]+0.08, tsne_data[i,1]+0.08), fontsize=6)
      plt.savefig('out/tsne.png', dpi=300)


 



if __name__ == "__main__":
  main()