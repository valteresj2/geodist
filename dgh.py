# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 23:51:30 2021

@author: valteresj
"""


import pandas as pd
import numpy as np
from tqdm import tqdm


def haversine_np(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
    
    
    
class dist_geo_haversine(object):
    def __init__(self,dt_ref=None,dt_comp=None,km=1,key=None,func=None,n_lat=None,n_long=None,path=None):
        if dt_ref is not None:
            self.dt_ref=dt_ref.copy()
        else:
            self.dt_ref=dt_ref
        if dt_comp is not None:
            self.dt_comp=dt_comp.copy()
        else:
            self.dt_comp=dt_comp
        self.km=km
        self.fun=func
        self.n_lat=n_lat
        self.n_long=n_long
        self.path=path
        self.func=func
        self.key=key
        
    
    def vizinhos_geo(self):
        lat_fil=self.km/100
        self.dt_ref=self.dt_ref.reset_index(drop=True)
        n=self.dt_ref.shape[0]
        cont=-1
        if self.dt_comp is not None:
            self.dt_comp=self.dt_comp.reset_index(drop=True)
            for i in tqdm(range(n)):
                index1=np.where(abs(self.dt_ref.loc[i,self.n_lat]-self.dt_comp[self.n_long].values)<lat_fil)[0]
                dist=haversine_np(np.array(self.dt_ref.loc[i,self.n_lat]),np.array(self.dt_ref.loc[i,self.n_long]),self.dt_comp.loc[index1,self.n_lat].values,self.dt_comp.loc[index1,self.n_long].values)
                index=np.where(dist<self.km)[0]
                index1=list(index1[index])
                if self.func is not None:   
                    new_f=pd.DataFrame(self.func(self.dt_comp.iloc[index1,:])).T
                    if (len(index)>0) & (self.path is not None):
                        self.dt_ref.loc[i,list(new_f.columns)]=list(new_f.values[0])
                        cont+=1
                        if cont==0:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_comp.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False)
                        else:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_comp.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False,header=False)
                    else:
                        self.dt_ref.loc[i,list(new_f.columns)]=[np.nan]*len(new_f.columns)
                    
                else:
                    if (len(index1)>0) & (self.path is not None) :
                        cont+=1
                        if cont==0:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_comp.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False)
                        else:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_comp.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False,header=False)
                    
                    
                    
        else:
            for i in tqdm(range(n)):
                index1=np.where(abs(self.dt_ref.loc[i,self.n_lat]-self.dt_ref[self.n_lat].values)<lat_fil)[0]
                dist=haversine_np(np.array(self.dt_ref.loc[i,self.n_lat]),np.array(self.dt_ref.loc[i,self.n_long]),self.dt_ref.loc[index1,self.n_lat].values,self.dt_ref.loc[index1,self.n_long].values)
                index=np.where(dist<self.km)[0]
                index1=list(index1[index])
                index1.remove(i)
                if self.func is not None:
                    new_f=pd.DataFrame(self.func(self.dt_ref.iloc[index1,:])).T
                    if len(index)>0:
                        self.dt_ref.loc[i,list(new_f.columns)]=list(new_f.values[0])
                        cont+=1
                        if cont==0:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_ref.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False)
                        else:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_ref.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False,header=False)
                    else:
                        self.dt_ref.loc[i,list(new_f.columns)]=[np.nan]*len(new_f.columns)
                else:
                    if len(index1)>0:
                        cont+=1
                        if cont==0:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_ref.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False)
                        else:
                            pd.DataFrame({self.key+'_ref':[self.dt_ref.loc[i,self.key]]*len(index1),self.key+'_comp':self.dt_ref.loc[index1,self.key]}).to_csv(self.path+'out_save.csv',sep=';',mode='a',index=False,header=False)
        return self.dt_ref
                
        
    
