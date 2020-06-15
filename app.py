import json

from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

#perform pca to get scree plot data
def perform_pca(data, columns, mode):
    # Reference https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
    scale_data = StandardScaler().fit_transform(data)
    
    pca = PCA()
    final_data = pca.fit_transform(scale_data)

    pvar = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = [x for x in range(1, len(pvar)+1)]
    
    pca_scree_data = pd.DataFrame(list(zip(labels, pvar)), columns=['PC_Number','Variance_Explained'])
    
    if mode ==0:
        pca_scree_data.to_csv('scree_plot_data_original.csv', index=False)
    elif mode == 1:
        pca_scree_data.to_csv('scree_plot_data_random.csv', index=False)
    elif mode ==2:
        pca_scree_data.to_csv('scree_plot_data_stratified.csv', index=False)
    else:
        print("Invalid mode")

#get top 3 attributes with highest pca loadings 
def get_top_pcal_attr(data, columns, mode):
    scale_data = StandardScaler().fit_transform(data)
    col_names = []

    for i in range(len(columns)):
        col_names.append('PC'+str(i+1))
    pca = PCA()
    final_data = pca.fit_transform(scale_data)
    loadings = pd.DataFrame(pca.components_.T, columns=col_names, index=columns)

    attr_loadings = []

    for i in range(len(columns)):
        # print("PC1")
       ss_val = loadings['PC1'][columns[i]]*loadings['PC1'][columns[i]] + loadings['PC2'][columns[i]]*loadings['PC2'][columns[i]]
       attr_loadings.append([columns[i], ss_val])

    # Driver Code 
    attr_loadings = Sort(attr_loadings)
    top_pca_attr_data = pd.DataFrame(attr_loadings[:3], columns=['Attribute_Name','PCA_Loading']) 

    d_three_list = list(zip(data[top_pca_attr_data['Attribute_Name'][0]], data[top_pca_attr_data['Attribute_Name'][1]], data[top_pca_attr_data['Attribute_Name'][2]]))
    data_top_three = pd.DataFrame(d_three_list, columns=[top_pca_attr_data['Attribute_Name'][0],top_pca_attr_data['Attribute_Name'][1], top_pca_attr_data['Attribute_Name'][2]])
    
    if mode ==0:
        top_pca_attr_data.to_csv('top_pca_attr_data_original.csv', index=False)
        data_top_three.to_csv('top_three_data_original.csv', index = False)
        
    elif mode == 1:
        top_pca_attr_data.to_csv('top_pca_attr_data_random.csv', index=False)
        data_top_three.to_csv('top_three_data_random.csv', index = False)
        
    elif mode ==2:
        top_pca_attr_data.to_csv('top_pca_attr_data_stratified.csv', index=False)
        data_top_three.to_csv('top_three_data_stratified.csv', index = False)

    else:
        print("Invalid mode")

#sort function used for getting top 3 attributes
def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[1], reverse = True) 
    return sub_li 

#Get pc1 and pc2 data loadings for the scatterplot
def get_top_two_pca_data(data, mode):
    scale_data = StandardScaler().fit_transform(data)
    
    pca = PCA()
    final_data = pca.fit_transform(scale_data)
    top_two_pca_data = pd.DataFrame(final_data[:, :2], columns=['PC1','PC2'])
    
    if mode ==0:
        top_two_pca_data.to_csv('top_two_pca_data_original.csv', index=False)
    elif mode == 1:
        top_two_pca_data.to_csv('top_two_pca_data_random.csv', index=False)
    elif mode ==2:
        top_two_pca_data.to_csv('top_two_pca_data_stratified.csv', index=False)
    else:
        print("Invalid mode")

def perform_mds(data, mode):
    embedding = MDS(n_components=2, dissimilarity= 'precomputed')
    
    data = preprocessing.scale(data)
    euclid_dist = pairwise_distances(data, metric = 'euclidean')
    corr_matrix = pairwise_distances(data, metric = 'correlation')
    
    euclid_mds = embedding.fit_transform(euclid_dist)
    euclid_mds_data = pd.DataFrame(euclid_mds, columns = ['MDS1', 'MDS2'])
    
    corr_mds = embedding.fit_transform(corr_matrix)
    corr_mds_data = pd.DataFrame(corr_mds, columns = ['MDS1', 'MDS2'])

    if mode ==0:
        euclid_mds_data.to_csv('euclidean_mds_data_original.csv', index=False)
        corr_mds_data.to_csv('correlation_mds_data_original.csv', index=False)
    elif mode == 1:
        euclid_mds_data.to_csv('euclidean_mds_data_random.csv', index=False)
        corr_mds_data.to_csv('correlation_mds_data_random.csv', index=False)
    elif mode ==2:
        euclid_mds_data.to_csv('euclidean_mds_data_stratified.csv', index=False)
        corr_mds_data.to_csv('correlation_mds_data_stratified.csv', index=False)
    else:
        print("Invalid mode")


app = Flask(__name__)
@app.route("/", methods = ['POST', 'GET'])
def index():
    global df
    #data samples
    data_orig = pd.read_csv('top_three_data_original.csv')
    data_rand = pd.read_csv('top_three_data_random.csv')
    data_strat = pd.read_csv('top_three_data_stratified.csv')
    #scree plot data
    data_orig_scree = pd.read_csv('scree_plot_data_original.csv')
    data_rand_scree = pd.read_csv('scree_plot_data_random.csv')
    data_strat_scree = pd.read_csv('scree_plot_data_stratified.csv')
    #top 2 attributes with highest PCA loading for each data
    attr_data_orig = pd.read_csv('top_pca_attr_data_original.csv')
    attr_data_rand = pd.read_csv('top_pca_attr_data_random.csv')
    attr_data_strat = pd.read_csv('top_pca_attr_data_stratified.csv')
    #PCA scatter data
    pca_scatter_data_orig = pd.read_csv('top_two_pca_data_original.csv')
    pca_scatter_data_rand = pd.read_csv('top_two_pca_data_random.csv')
    pca_scatter_data_strat = pd.read_csv('top_two_pca_data_stratified.csv')
    #mds scatter data
    mds_scatter_euclid_orig = pd.read_csv('euclidean_mds_data_original.csv')
    mds_scatter_corr_orig = pd.read_csv('correlation_mds_data_original.csv')
    mds_scatter_euclid_rand = pd.read_csv('euclidean_mds_data_random.csv')
    mds_scatter_corr_rand = pd.read_csv('correlation_mds_data_random.csv')
    mds_scatter_euclid_strat = pd.read_csv('euclidean_mds_data_stratified.csv')
    mds_scatter_corr_strat = pd.read_csv('correlation_mds_data_stratified.csv')
    
    #main data dictionary containing all plotting data
    data_dict = {}

    # Full Data
    chart_data = data_orig.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['original_data']= chart_data

    chart_data = data_rand.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['random_data']= chart_data

    chart_data = data_strat.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['strat_data']= chart_data

    #data for scree plot
    chart_data = data_orig_scree.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['original_data_scree']= chart_data

    chart_data = data_rand_scree.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['random_data_scree']= chart_data

    chart_data = data_strat_scree.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['strat_data_scree']= chart_data

    #data for top 3 attributes with highest pca loadings
    chart_data = attr_data_orig.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['original_data_attr']= chart_data

    chart_data = attr_data_rand.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['random_data_attr']= chart_data

    chart_data = attr_data_strat.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['strat_data_attr']= chart_data

    #data for PC1 and PC2 scatterplot
    chart_data = pca_scatter_data_orig.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['original_pca_scatter_data']= chart_data

    chart_data = pca_scatter_data_rand.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['random_pca_scatter_data']= chart_data

    chart_data = pca_scatter_data_strat.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['strat_pca_scatter_data']= chart_data

    #data for 2D MDS scatterplots (Euclidian & correlation distance)

    chart_data = mds_scatter_euclid_orig.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['original_mds_scatter_euclid']= chart_data

    chart_data = mds_scatter_corr_orig.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['original_mds_scatter_corr']= chart_data

    chart_data = mds_scatter_euclid_rand.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['random_mds_scatter_euclid']= chart_data

    chart_data = mds_scatter_corr_rand.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['random_mds_scatter_corr']= chart_data

    chart_data = mds_scatter_euclid_strat.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['strat_mds_scatter_euclid']= chart_data

    chart_data = mds_scatter_corr_strat.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data_dict['strat_mds_scatter_corr']= chart_data


    return render_template("assign2.html", data=data_dict)



if __name__ == "__main__":

    original_data = pd.read_csv('online_shoppers_intention_LabelEncode.csv', header = 0, error_bad_lines=False)
    original_data = original_data[:1000]

    # # RANDOM SAMPLING
    random_data = original_data.sample(frac = 0.25)
    random_data.to_csv('Online_shoppers_intention_random_sample.csv', index = False)
    random_data = pd.read_csv('Online_shoppers_intention_random_sample.csv')

    # # STRATIFIED SAMPLING
    # # STEP 1: Plot and Find Elbow:

    wcss = []
    for i in range(1, 5):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(original_data)
        wcss.append(kmeans.inertia_)

    # plt.plot(range(1, 5), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()

    # STEP 2: K-means clustering for optimal number of clusters =2
    clustering_kmeans = KMeans(n_clusters=2, precompute_distances="auto", n_jobs=-1)
    clustering_kmeans.fit(original_data)
    y = clustering_kmeans.fit_predict(original_data)
    original_data['clusters'] = y

    data_zeroc = original_data[original_data['clusters']==0]
    data_onec = original_data[original_data['clusters']==1]
    #for two clusters decide number of samples in each
    num_sample = int(len(original_data) * 0.25)//2
    if num_sample > len(data_zeroc):
      num_zeroc = len(data_zeroc)
      num_onec = 2*num_sample - len(data_zeroc)
    elif num_sample > len(data_onec):
      num_onec = len(data_onec)
      num_zeroc = 2*num_sample - len(data_onec)
    else:
      num_zeroc = num_sample
      num_onec = num_sample
    #get samples from each cluster and merge them to get final stratified sample
    data_zeroc = data_zeroc.sample(n= num_zeroc)
    data_onec = data_onec.sample(n= num_onec)

    strat_sample = [data_onec , data_zeroc]
    result_strat_sample = pd.concat(strat_sample, join= 'outer', axis=0)
    result_strat_sample = result_strat_sample.loc[:, result_strat_sample.columns != 'clusters']
    result_strat_sample.to_csv('Online_shoppers_intention_stratsampling.csv', index = False)
    original_data = original_data.loc[:, original_data.columns != 'clusters']

    strat_data = pd.read_csv('Online_shoppers_intention_stratsampling.csv')


    # #perform pca to get scree plot data for original data
    perform_pca(original_data, original_data.columns, 0)
    get_top_pcal_attr(original_data, original_data.columns, 0)
    get_top_two_pca_data(original_data, 0)
    # perform_mds(original_data, 0)
    
    # #perform pca to get scree plot data for random data
    perform_pca(random_data, random_data.columns, 1)
    get_top_pcal_attr(random_data, random_data.columns, 1)
    get_top_two_pca_data(random_data, 1)
    # perform_mds(random_data, 1)
    
    # #perform pca to get scree plot data for stratified data
    perform_pca(strat_data, strat_data.columns, 2)
    get_top_pcal_attr(strat_data, strat_data.columns, 2)
    get_top_two_pca_data(strat_data, 2)
    # perform_mds(strat_data, 2)

    
    app.run(debug=True)