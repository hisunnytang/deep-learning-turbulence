#!/home/kwoksun2/anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.cm as cm
import matplotlib

def reading_in_dict(filename):
    tf_predictions =  np.load(filename, encoding='latin1')
    beta  = np.array( tf_predictions.item().get('beta'))
    gamma = np.array(tf_predictions.item().get('gamma'))
    gamma_pred = np.array(tf_predictions.item().get('gamma_pred'))
    beta_pred  = np.array(tf_predictions.item().get('beta_pred'))
    
    return { "beta": beta, "gamma": gamma, "beta_pred": beta_pred, "gamma_pred": gamma_pred }

def calculate_loss( data_dict, parameter ):
    error              = data_dict[parameter] - data_dict[parameter + "_pred" ]
    percentage_error   = error  / data_dict[parameter]
    mean_squared_error = (error**2.0).mean()
    return error, percentage_error, mean_squared_error   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "filename",help = "the numpy array file to be plotted" )
    args = parser.parse_args()
    
    data_dict = reading_in_dict(args.filename)
    
    with plt.style.context("seaborn"):
        fig, axarr = plt.subplots(2,2)
        
        error_beta, perror_beta, mse_beta = calculate_loss( data_dict, 'beta' )
        axarr[0, 0].scatter( data_dict['beta'], perror_beta )
        axarr[0, 0].set_xlabel(r'$\beta$')
        axarr[0, 0].set_ylabel(r'fractional difference of $\beta$')
        axarr[0, 0].set_xlim( 0, 4.5 )
        axarr[0, 0].set_ylim( -1, 1)

        
        error_gamma, perror_gamma, mse_gamma = calculate_loss( data_dict, 'gamma' )
        axarr[0, 1].scatter( data_dict['gamma'], perror_gamma )
        axarr[0, 1].set_xlabel(r'$\gamma$')
        axarr[0, 1].set_ylabel('fractional difference of $\gamma$')
        axarr[0, 1].set_xlim(0, 4)
        axarr[0, 1].set_ylim(-1,1)        


        axarr[1, 0].scatter( perror_gamma, perror_beta )
        axarr[1, 0].set_xlabel(r'fractional difference of $\gamma$')
        axarr[1, 0].set_ylabel(r'fractional difference of $\beta$')
        axarr[1, 0].set_xlim(-1, 1)
        axarr[1, 0].set_ylim(-1,1) 

        error = error_gamma**2.0 + error_beta**2.0        
        minima = min(error)
        maxima = max(error)
        norm = matplotlib.colors.Normalize(vmin=minima, vmax= maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.magma)
         
        color_error = [ mapper.to_rgba(x) for x in error ]
        axarr[1,1].quiver( data_dict['gamma'], data_dict['beta'], 
                           data_dict['gamma_pred'] - data_dict['gamma'],
                           data_dict['beta_pred']  - data_dict['beta'], color = color_error )
        axarr[1,1].set_xlabel(r'$\gamma$')
        axarr[1,1].set_ylabel(r'$\beta$')
        axarr[1,1].set_xlim(-0.5,3.5)
        axarr[1,1].set_ylim(1,4.5) 
        plt.tight_layout()
        fig.savefig( args.filename.strip('.npy'+".png" )) 
