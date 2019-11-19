# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams



"""
COLUMN HEADER NAMES:
    (Company Screening Reports)
    
NOE = Number of Employees - Global (Latest)	
EBITDA = EBITDA [LTM] ($USDmm, Historical rate)	
R_D = R&D Expense [LTM] ($USDmm, Historical rate)	
TEV = Total Enterprise Value [My Setting] [Latest] ($USDmm, Historical rate)	
EPS = Basic EPS [LTM] ($USD, Historical rate)	
COE = Cost Of Revenues [LTM] ($USDmm, Historical rate)	
TOE = Total Operating Expenses [LTM] ($USDmm, Historical rate)
"""

cols = ['NOE', 'EBITDA', 'R_D', 'TEV', 'EPS', 'COE', 'TOE']
energy_SD_df = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningEnergy.xlsx', names=cols)
pharm_SD_df = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningPharmaceutical.xlsx', names=cols)
ss_SD_df = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningSoftwareAndServices.xlsx', names=cols)
#ss_SD_df2 = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningSoftwareAndServices2.xlsx', names=cols)

ind_SD_df= pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningIndustrials.xlsx', names=cols)
health_SD_df = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningHealthCare.xlsx', names=cols)
fin_SD_df = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningFinancials.xlsx', names=cols)
it_SD_df = pd.read_excel('C:/Users/aaron/OneDrive/Documents/U of U/2019 Spring/CS5140/Project/Datasets/ScreeningInformationTechnology.xlsx', names=cols)

def clean_empty_data(df, industry):
    df = df.applymap(str)
    df = df.drop(df.index[df.NOE == '-'])
    df = df.drop(df.index[df.EBITDA == '-'])
    df = df.drop(df.index[df.R_D == '-'])
    df = df.drop(df.index[df.TEV == '-'])
    df = df.drop(df.index[df.EPS == '-'])
    df = df.drop(df.index[df.COE == '-'])
    df = df.drop(df.index[df.TOE == '-'])

    return df.values.astype(float)

def binDataByCapSize(listOfValues):
    listOfValues = np.array(listOfValues)
    row, col = listOfValues.shape
    
    listInOrder = list()
    for i in range(row):
        if (listOfValues[i][3]*1_000_000 < 500_000):
            listInOrder.append("smallCap")
        elif(listOfValues[i][3]*1_000_000 >= 500_000 and listOfValues[i][3]*1_000_000 < 2_000_000_000):
            listInOrder.append("midCap")
        else:
            listInOrder.append("largeCap")
    
    return listInOrder


SD_np_dict = dict()

it_SD_np = clean_empty_data(it_SD_df, "Information Technology")
ss_SD_np = clean_empty_data(ss_SD_df, "Software")
#ss_SD_np2 = clean_empty_data(ss_SD_df2, "Software2")

fin_SD_np = clean_empty_data(fin_SD_df, "Industry")

health_SD_np = clean_empty_data(health_SD_df, "Health")
pharm_SD_np = clean_empty_data(pharm_SD_df, "Pharmacy")

ind_SD_np = clean_empty_data(ind_SD_df, "Industry")
energy_SD_np = clean_empty_data(energy_SD_df, "Energy")

SD_np_dict["Information Technology"] = (it_SD_np)
SD_np_dict["Health"] = (health_SD_np)
SD_np_dict["Industry"] = (ind_SD_np)

allData = []
allIndustries = list()
allIndustries += ["Information Technology"] * len(SD_np_dict["Information Technology"])
allData.extend(SD_np_dict["Information Technology"])
allIndustries += ["Health"] * len(SD_np_dict["Health"])
allData.extend(SD_np_dict["Health"])
allIndustries += ["Industry"] * len(SD_np_dict["Industry"])
allData.extend(SD_np_dict["Industry"])

def plotLoadPC(df, coeff,binType,plotType=None, targets = None,plotTitle = None, labels=None, fc=0, sc=1, tc=2):
    n = coeff.shape[0]
    colors = ['r', 'g', 'b']
    
    ######################2D
    if plotType is None:
        for target, color in zip(targets,colors):
            indicesToKeep = df[binType] == target
            plt.scatter(df.loc[indicesToKeep, 'PC' + str(fc+1)]
                       , df.loc[indicesToKeep, 'PC' + str(sc+1)]
                       , c = color
                       , s = 0.5)
            
        for i in range(n):
            plt.arrow(0, 0, coeff[i,fc], coeff[i,sc],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,fc] * 1.15, coeff[i,sc] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,fc] * 1.15, coeff[i,sc] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        
        plt.xlabel("Principal Component " + str(fc))
        plt.ylabel("Principal Component " + str(sc))
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
    #####################3D
    else:
        fig= plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for target, color in zip(targets,colors):
            indicesToKeep = df[binType] == target
            ax.scatter(xs=df.loc[indicesToKeep, 'PC' + str(fc+1)], ys=df.loc[indicesToKeep, 'PC' + str(sc+1)], zs=df.loc[indicesToKeep, 'PC' + str(tc+1)] )
        
        
        

        for i in range(n):
            ax.plot(coeff[0:i,fc], coeff[0:i,sc], coeff[0:i,tc])
    #        plt.arrow(0, 0, coeff[i,fc], coeff[i,sc],color = 'r',alpha = 0.5)
            if labels is None:
                ax.text(coeff[i,fc], coeff[i,sc], coeff[i,tc] + 2, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                ax.text(coeff[i,fc], coeff[i,sc], coeff[i,tc] + 2, labels[i], color = 'g', ha = 'center', va = 'center')
        
        ax.set_xlabel('\nPC1')
        ax.set_ylabel('\nPC2')
        ax.set_zlabel('\nPC3')
    
        ax.dist = 12
    
        df = np.array(df)
        rcParams['axes.labelpad'] = 10

        ax.set_xticks(np.arange(min(df[:,fc]), max(df[:,fc])+1, 10))
        ax.set_yticks(np.arange(min(df[:,sc]), max(df[:,sc])+1, 10))
        ax.set_zticks(np.arange(min(df[:,tc]), max(df[:,tc])+1, 10))
    
    if plotTitle is not None:
        plt.title(plotTitle)

    plt.legend(targets)
    plt.grid()
    plt.show()
    
def pca_cor(binType, dataInList = None, allIndustries=None, dictionary=None, key=None, verbose=True, graph=True, width=10, height=10):
    labels = ['NOE', 'EBITDA', 'R_D', 'TEV', 'EPS', 'COE', 'TOE']
    if binType == "Cap":
        targets = ['smallCap', 'midCap', 'largeCap']
    elif binType == "Industry":
        targets = ['Information Technology', 'Health', 'Industry']
        
    if (dictionary is not None and key is not None):
        data_std = StandardScaler().fit_transform(np.array(SD_np_dict[key]))
    elif (dataInList is not None):
        data_std = StandardScaler().fit_transform(np.array(dataInList))

    pca = PCA()
    value = pca.fit_transform(data_std)    
    pca_components = np.transpose(pca.components_)

    cor_mat1 = np.corrcoef(data_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    idx = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    
    principalDf = pd.DataFrame(data=value, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])    
    if dataInList is not None:
        binnedData = binDataByCapSize(dataInList)
        principalDf["Cap"] = binnedData
        plotTitle="Loading Plot using variables \n'NOE', 'EBITDA', 'R_D', 'TEV', 'EPS', 'COE', and 'TOE'"
    elif dictionary is not None and key is not None:
        principalDf["Cap"] = binDataByCapSize(dictionary[key])
        plotTitle = "Loading Plot on Dataset: " + key
    if allIndustries is not None:
        if dataInList is not None:
            principalDf["Industry"] = allIndustries
        elif dictionary is not None and key is not None:
            principalDf["Industry"] = allIndustries
        
    if graph:
        plt.plot(sorted(eig_vals, reverse=True))
        if (key is not None):
            plt.title("Scree Plot on " + key)
        else:
            plt.title("Scree Plot on All Data")

        plt.show()
        
        #Call the function. Use only the 2 PCs.
        for i in range(len(labels)):
            for j in range(i, 3):
                if (i != j):
                    #Eigenvectors
                    plotLoadPC(principalDf, pca_components, binType, targets=targets, plotTitle=plotTitle, labels=labels, fc=i, sc=j)
#                    plotLoadPC(principalDf, eig_vecs, binType, targets=targets, plotTitle="Loading Plot using variables \n'NOE', 'EBITDA', 'R_D', 'TEV', 'EPS', 'COE', and 'TOE'", labels=labels, fc=i, sc=j)
    
    if verbose:
        print()
        print('Eigenvectors \n%s' %eig_vecs)
        print('\nEigenvalues \n%s' %eig_vals)
        print('\nVariance Ratio \n%s' %pca.explained_variance_ratio_)
        
    if dictionary is not None and key is not None:
        ogDf = pd.DataFrame(data=dictionary[key], columns=labels)
        ogDf.reset_index(drop=True, inplace=True)
        principalDf.reset_index(drop=True, inplace=True)
        frames = [principalDf, ogDf]
        finalDf = pd.concat(frames,sort=False,axis=1)
    elif dataInList is not None:
        ogDf = pd.DataFrame(data=dataInList, columns=labels)
        ogDf.reset_index(drop=True, inplace=True)
        principalDf.reset_index(drop=True, inplace=True)
        frames = [principalDf, ogDf]
        finalDf = pd.concat(frames,sort=False,axis=1)
     
    return finalDf

'''Currently supports only up to 7 clusters for differing graph colors'''
def graph_kmeans(dataframe, name, verbose=False, xMinTick = None, xMaxTick=None, yMinTick = None, yMaxTick=None, cluster_count=7, width=10, height=10):
    km = KMeans(n_clusters=cluster_count).fit(dataframe)
    dataframe['cluster'] = km.labels_
    
    if verbose:
        print("K-Means for " + name)
    
    if width is not None and height is not None:
        plt.figure(figsize=(width,height))
    
    dataframe = np.array(dataframe)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    clusters = dict()
    for i in range(0, dataframe.shape[0]):
        if km.labels_[i] == 0:
            c1 = plt.scatter(dataframe[i,0],dataframe[i,1],c='r', marker='+')
            clusters["Cluster 1"] = c1
        elif km.labels_[i] == 1:
            c2 = plt.scatter(dataframe[i,0],dataframe[i,1],c='g', marker='o')
            clusters["Cluster 2"] = c2
        elif km.labels_[i] == 2:
            c3 = plt.scatter(dataframe[i,0],dataframe[i,1],c='b', marker='v')
            clusters["Cluster 3"] = c3
        elif km.labels_[i] == 3:
            c4 = plt.scatter(dataframe[i,0],dataframe[i,1],c=colors["blueviolet"], marker='*')
            clusters["Cluster 4"] = c4
        elif km.labels_[i] == 4:
            c5 = plt.scatter(dataframe[i,0],dataframe[i,1],c=colors["orchid"], marker='s')
            clusters["Cluster 5"] = c5
        elif km.labels_[i] == 5:
            c6 = plt.scatter(dataframe[i,0],dataframe[i,1],c=colors["brown"], marker='h')
            clusters["Cluster 6"] = c6
        elif km.labels_[i] == 6:
            c7 = plt.scatter(dataframe[i,0],dataframe[i,1],c=colors["gold"], marker='H')
            clusters["Cluster 7"] = c7
            
    if xMinTick is None and xMaxTick is None:
        plt.xlim(np.arange(min(dataframe[:,0]), max(dataframe[:,0])+1, 1))
    else:
        if (xMaxTick is not None and xMinTick is not None):
            plt.xlim(xMinTick, xMaxTick)
        elif (xMinTick is not None):
            plt.xlim(xMinTick, max(dataframe[:,0])+1)
        elif (xMaxTick is not None):
            plt.xlim(min(dataframe[:,0]), xMaxTick)

    if yMinTick is None and yMaxTick is None:
        plt.ylim(np.arange(min(dataframe[:,0]), max(dataframe[:,0])+1, 1))
    else:
        if (yMaxTick is not None and yMinTick is not None):
            plt.ylim(yMinTick, yMaxTick)
        elif (yMinTick is not None):
            plt.ylim(yMinTick, max(dataframe[:,1])+1)
        elif (yMaxTick is not None):
            plt.ylim(min(dataframe[:,1]), yMaxTick)
            
    plt.legend(list(clusters.values()),list(clusters.keys()))
    plt.grid(True)
    
    plt.title('K-Means Clustering on ' + name)
    plt.show()
    
    dataframe = pd.DataFrame(data=dataframe, columns=["PC1", "PC2", "Cluster"])
    return dataframe
    
    
def dbscan(np_array, name, set_eps=5, xMinTick = None, xMaxTick=None, yMinTick = None, yMaxTick=None):
    clustering = DBSCAN(eps=set_eps).fit(np_array)
    
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
        
        xy = np_array[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=col, markersize=3)
    
        xy = np_array[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=col, markersize=1)
        
    
#    if xMinTick is None and xMaxTick is None:
##        plt.xlim(np.arange(min(xy[:,0]), max(xy[:,0])+1, 1))
#    else:
#        if (xMaxTick is not None and xMinTick is not None):
#            plt.xlim(xMinTick, xMaxTick)
#        elif (xMinTick is not None):
#            plt.xlim(xMinTick, max(xy[:,0])+1)
#        elif (xMaxTick is not None):
#            plt.xlim(min(xy[:,0]), xMaxTick)
#
#    if yMinTick is None and yMaxTick is None:
#        plt.ylim(np.arange(min(xy[:,0]), max(xy[:,0])+1, 1))
#    else:
#        if (yMaxTick is not None and yMinTick is not None):
#            plt.ylim(yMinTick, yMaxTick)
#        elif (yMinTick is not None):
#            plt.ylim(yMinTick, max(xy[:,1])+1)
#        elif (yMaxTick is not None):
#            plt.ylim(min(xy[:,1]), yMaxTick)
        
    plt.xlim(-1, 10)
    plt.ylim(-1, 10)
    plt.grid(True)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    
    df = pd.DataFrame(data=np_array, columns= ['PC1', 'PC2'])
    df['Cluster'] = clustering.labels_
    
    return df


#pca_cor("Cap", dataInList=allData, allIndustries=allIndustries, verbose=True)
all_data = pca_cor("Industry", dataInList=allData, allIndustries=allIndustries, verbose=False, graph=False)

#pca_cor("Cap", dictionary=SD_np_dict, key="Information Technology", verbose = True)

#pca_cor("Cap", dictionary=SD_np_dict, key="Health",verbose = True)

#pca_cor("Cap", dictionary=SD_np_dict, key="Industry",verbose = False)

#for key, value in SD_np_dict.items():
#    graph_kmeans(value, key)

#for key,value in SD_np_dict.items():
#    dbscan(value, key, set_eps=5000)


#all_data_frame = graph_kmeans(all_data.iloc[:,0:2], "All Data", cluster_count = 2, xMinTick=-1, xMaxTick=10, yMinTick=-1, yMaxTick=10)
#
#frames = [all_data, all_data_frame["Cluster"]]
#all_data = pd.concat(frames, axis=1,sort=False)
#
#all_data_filtered = all_data[all_data['Cluster'] == 0.0]
#allIndustries2 = np.array(all_data_filtered["Industry"])
#all_data_filtered = all_data_filtered.drop(['Cap', 'Industry', 'Cluster','PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], axis=1)
#
#all_data_second_pass = pca_cor("Industry", dataInList=all_data_filtered, allIndustries=allIndustries2, verbose=True, graph=True)

all_data_np = np.array(all_data.iloc[:,0:2])
all_data_filtered_dbscan = dbscan(all_data_np, "All Data", set_eps = 0.15)

all_data = all_data.drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], axis=1)
frames = [all_data, all_data_filtered_dbscan["Cluster"]]
all_data = pd.concat(frames, axis=1,sort=False)
all_data = all_data.loc[all_data['Cluster'] == 0]
print(all_data)
all_data.to_excel("dbscanFiltered.xlsx")



