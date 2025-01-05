
import yfinance as yf
import numpy as np
import sys
sys.path.append('..')
sys.path.append('src')
from preprocessing import Preprocessing
import matplotlib.pyplot as plt
from model import ExtremeSphericalKMeans

class clustering():

    def __init__(self, tickers, data, nb_clusters, year, THRESHOLD=0.1):

        self.tickers = tickers
        self.data =  np.asarray(data)
        self.sector_info = {}
        self.extreme_data = Preprocessing.transform_to_extreme_values(data)
        self.largest_data, self.retained_indices = Preprocessing.process(self.extreme_data)
        self.nb_clusters = nb_clusters
        self.year = year
        self.THRESHOLD = THRESHOLD
    
    def get_sector_info(self):
        """ 
        Compute the sector of each ticker 
        """

        # Get the sector for each tickers
        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info
                self.sector_info[ticker] = info.get("sector", "N/A")  
            except Exception as e:
                self.sector_info[ticker] = f"Error: {e}"



    def get_labels(self):
        """
        Predict the closest cluster each sample in data belongs to. 

        Returns 
        -------
        list
        A list containing the labels of the cluster to which each data point belongs.
        """

        kmeans = ExtremeSphericalKMeans(n_clusters=self.nb_clusters, max_iter=10, threshold=self.THRESHOLD)
        kmeans.fit(self.data)
        labels = kmeans.predict(self.data)

        return labels
    

    def get_percentage_industries(self):
        """ 
        Computes for each cluster, the percentage of each sector.
         
        Returns
        ------- 
        dict
        A dictionnary containing all percentage of each sector for each cluster
          """
        
        self.get_sector_info()

        list_industries = set(self.sector_info.values())
        labels = self.get_labels()
        dict_percentage = {key: [] for key in list_industries}

        for i in range(self.nb_clusters):
            count = {key: [] for key in list_industries}
            values_list = list(self.sector_info.values())
            indices = np.where(labels == i)[0]

            all_sectors_in_cluster_i = [values_list[idx] for idx in self.retained_indices]
            all_sectors_in_cluster_i = [all_sectors_in_cluster_i[idx] for idx in indices]

            for sector in list_industries :

                count[sector] = all_sectors_in_cluster_i.count(sector)
                dict_percentage[sector].append(count[sector])

        return dict_percentage
    

    def plot_pie_for_each_cluster(self):
        """
        Plot a pie chart for each cluster, showing the percentage representation of each sector.
        """

        _, ax = plt.subplots(1, self.nb_clusters, figsize=(22, self.nb_clusters + 5))

        dict_percentage = self.get_percentage_industries()

        for k in range(self.nb_clusters):
            percentage_cluster_k = [percentages[k] for percentages in dict_percentage.values() if percentages[k] != 0]
            all_perc_cluster_k = [percentages[k] for percentages in dict_percentage.values()]

            sectors = list(dict_percentage.keys())
            legend = [sectors[k] for k, value in enumerate(all_perc_cluster_k) if value != 0]

            ax[k].pie(percentage_cluster_k, labels =legend, normalize = True, labeldistance = 1.3, autopct = lambda x: str(round(x, 2)) + '%')
            ax[k].set_title('Cluster {}'.format(k+1))

        plt.suptitle('Clusters for {}'.format(self.year))

