######################
# MCMC JAM model
######################
#...Import packages
from scipy.io.idl import readsav
import dyn_py
from dyn_py.jam_axi_rms import jam_axi_rms
from dyn_py.mge_vcirc import mge_vcirc
from dyn_py.readcol import readcol


from pdb import set_trace as stop
import emcee
import math
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.stats as ss
import scipy.interpolate as interp
from astropy.table import Table
from scipy.stats.mstats import mode
import triangle




def rms_logprob(p,surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot, \
                       Mbh, dist, xmod, ymod, Vrmsbin, dVrmsbin,beta_in,ml_in,ideg_in,ideg_dwn):

    
    beta_scalar=p[0]
    ml=p[1]
    ideg=p[2]
    
    


    priors =  ss.uniform.logpdf(beta_scalar,loc=-1.0,scale=2)+\
              ss.uniform.logpdf(ml,loc=0.0,scale=10.0)+\
              ss.uniform.logpdf(ideg,loc=ideg_dwn,scale=90.0)+\
              ss.norm.logpdf(beta_scalar,loc=beta_in,scale=0.39)+\
              ss.norm.logpdf(ml,loc=ml_in,scale=0.24)+\
              ss.norm.logpdf(ideg,loc=ideg_in,scale=2.0)
              
               
              
    
    if np.isfinite(priors) == False:
        return -np.inf
      
    if (np.cos(ideg*math.pi/180))**2 >=  (qmin**2):
       return -np.inf 
     
      
    inc = np.radians(ideg)
    qintr_lum = qobs_lum**2 - np.cos(inc)**2 
    if np.any(qintr_lum <= 0):
       return -np.inf   
    
    qintr_lum = qobs_lum**2 - np.cos(inc)**2 
    qintr_lum = np.sqrt(qintr_lum)/np.sin(inc)
    if np.any(qintr_lum < 0.05):
       return -np.inf      
    
    qintr_pot = qobs_pot**2 - np.cos(inc)**2
    if np.any(qintr_pot <= 0):
       return -np.inf  
    
    qintr_pot = np.sqrt(qintr_pot)/np.sin(inc)
    if np.any(qintr_pot < 0.05):
       return -np.inf  
     

    #print 'ideg=',ideg, 'cos2=', (np.cos(ideg*math.pi/180))**2,  'q2=',  (qmin**2)
    
    #...Call JAM model
    rmsModel, _, chi2,_ = \
    jam_axi_rms(surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot,
                     ideg, Mbh, dist, xmod, ymod, plot=True, rms=Vrmsbin,ml=ml, erms=dVrmsbin,
                      sigmapsf=0.6, beta=beta_scalar+(surf_lum*0), pixsize=0.8)
    
    lp = -chi2*(len(Vrmsbin)-3) + priors
    print(lp)
    if np.isnan(lp):
        return -np.inf

    return lp
   
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# MCMC for Vobs and sigma_obs
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

gals=['488','772','4102','5678','3949','4030','2964','628','864','4254','1042','3346','3423','4487','2805','4775','5585','5668']

qmin_all=[0.77,0.66,0.555,0.525,0.64,0.76,0.55,0.81,0.68,0.73,0.71,0.84,0.77,0.63,0.76,0.865,0.64,0.845]
ideg_dwn_all=[39.65,48.75,56.3,58.35,50.25,40.55,56.65,35.95,47.2,43.15,44.8,32.9,39.65,50.95,40.55,30.15,50.25,32.35]

Vsys_all=[2299,2506,838,1896,808,1443,1324,703,1606,2384,1404,1257,1001,1016,1742,1547,312,1569]

beta_all=[-0.10,0.0,0.40,0.40,0.60,0.0,0.60,0.0,0.60,0.10,0.10,0.0,-0.10,0.50,0.30,0.50,0.60,0.60]
ml_all=[1.19,0.98,0.63,1.20,1.43,0.74,1.37,1.11,1.60,0.49,1.95,2.02,2.36,2.12,2.05,1.03,3.46,1.41]
ideg_all=[40,49,57,59,51,41,57,36,48,44,45,42,40,54,41,34,51,37]


for j in range(0,len(gals),1):
    
  gal=gals[j]
  print 'GAL=',gal
  qmin=qmin_all[j]
  ideg_dwn=ideg_dwn_all[j]
  Vsys=Vsys_all[j]
  
  beta_in=beta_all[j]
  ml_in=ml_all[j]
  ideg_in=ideg_all[j]
  
  
  
    #...Read sav-files
  props = readsav("data_input/SAV_files/NGC"+gal+"_axi_rms.sav") # IDL .sav or .idl file
  surf_lum=props["surf_lum"]
  sigobs_lum=props["sigobs_lum"]
  qobs_lum=props["qobs_lum"]
  surf_pot=props["surf_pot"]
  sigobs_pot=props["sigobs_pot"]
  qobs_pot=props["qobs_pot"]
  #ideg=props["ideg"]
  Mbh=props["Mbh"]
  dist=props["dist"]
  xmod=props["xmod"]
  ymod=props["ymod"]
  #Vrmsbin=props["Vrmsbin"]
  #beta_vec=props["beta_vec"]
  flux_bin=props["fluxbin"]
  #dVrmsbin=0.05*Vrmsbin


  hlist = fits.open('data_input/stellar_kinematics/PXF_bin_MS_NGC'+gal+'_r1_MILESstars_SN60.fits')
  tab=hlist[1].data
  xbin = tab['xs']
  ybin = tab['ys']
  Vbin = tab['vpxf'] -Vsys # subtract systemic velocity 
  Sbin = tab['spxf']
  dVbin = tab['dvp']
  dSbin = tab['dsp']

  Vrmsbin=np.sqrt(Vbin**2+Sbin**2)
  dVrmsbin=np.sqrt((dVbin*Vbin)**2 + (dSbin*Sbin)**2)/Vrmsbin
  

  # Set the walkers
  ndim, nwalkers = 3,20
  p0 = np.zeros((nwalkers,ndim))
  p0[:,0] = np.random.randn(nwalkers)*0.05+beta_in
  p0[:,1] = np.random.randn(nwalkers)*0.05+ml_in
  start_ideg = np.random.randn(nwalkers)*1.0+ideg_in
  start_ideg[start_ideg < ideg_dwn]=ideg_in
  p0[:,2] = start_ideg

  sampler = emcee.EnsembleSampler(nwalkers, ndim, rms_logprob,
                                  args=[surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot, \
                                        Mbh, dist, xmod, ymod, Vrmsbin, dVrmsbin, beta_in, ml_in, \
                                        ideg_in, ideg_dwn], threads=4)


  pos, prob, state = sampler.run_mcmc(p0, 60)
  #------------------------------------------
  fig = plt.figure(figsize=(10,6))
  plt.subplot(3,2,1)
  plt.title(r'NGC'+gal)
  plt.plot(sampler.chain[:,:,0].T)
  plt.ylabel(r'Chain for $\beta_z$')
  plt.subplot(3,2,2)
  plt.plot(sampler.chain[:,:,1].T)
  plt.ylabel(r'Chain for $\Upsilon$')
  plt.subplot(3,2,3)
  plt.plot(sampler.chain[:,:,2].T)
  plt.ylabel(r'Chain for $ideg$')
  #plt.subplot(3,2,4)
  #plt.plot(sampler.chain[:,:,3].T)
  #plt.ylabel(r'Chain for $dVrms$')
  #   
  plt.tight_layout() # This tightens up the spacing
  plt.savefig("figures/chains/NGC"+gal+"_JAM_chain.pdf")
  plt.close()

  #... save the data in a file
  np.savez('data_output/chains/NGC'+gal+'_chain', chain_JAM=sampler.chain, lnprobability_JAM=sampler.lnprobability)
      
  #------------------------------------------------

  sampler.reset()
  pos,prob,state = sampler.run_mcmc(pos,60)
  #-----------------------------------------------
  fig = plt.figure(figsize=(10,6))
  triangle.corner(sampler.flatchain, labels=[r"$\beta_z$", r"$\Upsilon$",r"$i (^{\circ})$"],bins=15)
  fig.gca().annotate("NGC", xy=(0.5, 1.0), xycoords="figure fraction",xytext=(0, -5), textcoords="offset points",ha="center", va="top")
  plt.savefig("figures/triangles/NGC"+gal+"_JAM_triangle.pdf")
  plt.close()
   
  #... save the data in a file
  np.savez('data_output/chains/NGC'+gal+'_flatchain', flatchain_JAM=sampler.flatchain, flatlnprobability_JAM=sampler.flatlnprobability)
  #-----------------------------------------------


  beta_dist = sampler.flatchain[:,0]
  beta_md=np.median(beta_dist)
  beta_plus=np.percentile(sampler.flatchain[:,0], 75)- np.median(sampler.flatchain[:,0])
  beta_minus=np.median(sampler.flatchain[:,0]) - np.percentile(sampler.flatchain[:,0], 25)

  ml_dist = sampler.flatchain[:,1]
  ml_md=np.median(ml_dist)
  ml_plus=np.percentile(sampler.flatchain[:,1], 75)- np.median(sampler.flatchain[:,1])
  ml_minus=np.median(sampler.flatchain[:,1]) - np.percentile(sampler.flatchain[:,1], 25)
  
  ideg_dist = sampler.flatchain[:,2]
  ideg_md=np.median(ideg_dist)
  ideg_plus=np.percentile(sampler.flatchain[:,2], 75)- np.median(sampler.flatchain[:,2])
  ideg_minus=np.median(sampler.flatchain[:,2]) - np.percentile(sampler.flatchain[:,2], 25)
  #-----------------------------------------------
  # Array for the medians
  filename='data_output/Tables/Table_NGC_'+gal+'.txt'
  medians = [beta_md, ml_md, ideg_md]

  # Array for the upper percentiles
  ups = [beta_plus, ml_plus, ideg_plus]

  # Array for the lower percentiles
  lws = [beta_minus, ml_minus, ideg_minus]

  t = Table([medians, ups, lws], names = ('medians','ups','lws'))
  t.write(filename, format = 'ascii')
  #-------------------------------------------------
  print 'Beta_z:' , np.median(sampler.flatchain[:,0]), \
      '+', np.percentile(sampler.flatchain[:,0], 75) - np.median(sampler.flatchain[:,0]),\
      '-', np.median(sampler.flatchain[:,0]) - np.percentile(sampler.flatchain[:,0], 25) 
  print 'M/L: ', np.median(sampler.flatchain[:,1]), \
      '+', np.percentile(sampler.flatchain[:,1], 75) - np.median(sampler.flatchain[:,1]),\
      '-', np.median(sampler.flatchain[:,1]) - np.percentile(sampler.flatchain[:,1], 25) 
  print 'Incl: ', np.median(sampler.flatchain[:,2]), \
      '+', np.percentile(sampler.flatchain[:,2], 75) - np.median(sampler.flatchain[:,2]),\
      '-', np.median(sampler.flatchain[:,2]) - np.percentile(sampler.flatchain[:,2], 25)
  #---------------------------------------------------------------
  bins_rms = 30

  fig = plt.figure(figsize=(18,10))
  plt.subplot(3,3,1)
  patches = plt.hist(sampler.flatchain[:,0],bins=bins_rms)
  plt.plot((np.median(beta_dist),np.median(beta_dist)), (0,160), color = 'r')
  plt.title(r'Distribution of $Beta_z$')

  plt.subplot(3,3,2)
  patches = plt.hist(sampler.flatchain[:,1],bins=bins_rms)
  plt.plot((np.median(ml_dist),np.median(ml_dist)), (0,160), color = 'r')
  plt.title(r'Distribution of  M/L$')

  plt.subplot(3,3,3)
  patches = plt.hist(sampler.flatchain[:,2],bins=bins_rms)
  plt.plot((np.median(ideg_dist),np.median(ideg_dist)), (0,160), color = 'r')
  plt.title(r'Distribution of ideg$')

  plt.savefig("figures/NGC"+gal+"/NGC"+gal+"_MCMC_dist.pdf")
  plt.close()


  #... save the data in a file
  np.savez('data_output/NGC'+gal+'/NGC'+gal+'_distributions', beta_dist=beta_dist, ml_dist=ml_dist, ideg_dist=ideg_dist)

  data = np.load('data_output/NGC'+gal+'/NGC'+gal+'_distributions.npz')
  beta_dist=data['beta_dist']
  ml_dist=data['ml_dist']
  ideg_dist=data['ideg_dist']

  

  #########################################
  # CIRCULAR VELOCITY
  ########################################
    #...Read sav-files
  props = readsav("data_input/SAV_files/NGC"+gal+"_axi_rms.sav") # IDL .sav or .idl file
  surf_lum=props["surf_lum"]
  sigobs_lum=props["sigobs_lum"]
  qobs_lum=props["qobs_lum"]
  Mbh=props["Mbh"]
  dist=props["dist"]



  rad, _, _, _, _, vobs, evobs, _, _, _ = readcol('data_input/Vcirc_radius/Fourier_Vbin_'+gal+'.txt', skipline = 5, twod=False)
  #rad = np.logspace(-1,2,25)

  beta_dist2=beta_dist
  ml_dist2=ml_dist
  ideg_dist2=ideg_dist
  
  

  #beta_dist2=beta_dist[((np.cos(np.radians(ideg_dist)))**2 < (qmin**2))] 
  #ml_dist2=ml_dist[((np.cos(np.radians(ideg_dist)))**2 < (qmin**2))] 
  #ideg_dist2=ideg_dist[((np.cos(np.radians(ideg_dist)))**2 < (qmin**2))] 

  #beta_dist2=beta_dist[(np.cos(np.radians(ideg_dist)))**2 < (qobs_lum**2)] 
  #ml_dist2=ml_dist[(np.cos(np.radians(ideg_dist)))**2 < (qobs_lum**2)] 
  #ideg_dist2=ideg_dist[(np.cos(np.radians(ideg_dist)))**2 < (qobs_lum**2)] 

  vcircs = np.zeros([len(rad),len(ideg_dist2)])


  for i in range(len(ml_dist2)):
    
    ml= ml_dist2[i]
    ideg = ideg_dist2[i]
    
    
    
    
    
    #condition
    inc = np.radians(ideg)      # ...convert inclination to radians
    qintr = qobs_lum**2 - np.cos(inc)**2
    
    w=np.where(  (qintr > 0.0)  &  (np.sqrt(qintr)/np.sin(inc) > 0.05)  )
    
    
    
    qobs_lum=qobs_lum[w] 
    surf_lum=surf_lum[w] 
    sigobs_lum=sigobs_lum[w] 
    dummy = mge_vcirc(surf_lum*ml, sigobs_lum, qobs_lum, ideg, Mbh, dist, rad)
    #stop()
    vcircs[:,i] = dummy


  vcirc_med = np.median(vcircs, axis = 1)
  vcirc_up = np.percentile(vcircs, 75, axis = 1) - np.median(vcircs, axis = 1)
  vcirc_dn = np.median(vcircs, axis = 1) - np.percentile(vcircs, 25, axis = 1)

  #...Calculate Vcirc
  #rad = np.logspace(-1,2,25) # Radii in arscec where Vcirc has to be computed
  #vcirc = mge_vcirc(surf_lum*ml, sigobs_lum, qobs_lum, ideg, Mbh, dist, rad)

  #...Plot Vcirc    
  plt.clf()
  plt.plot(rad, vcirc_med, '-o')
  plt.errorbar(rad,vcirc_med, yerr = (vcirc_up, vcirc_dn), color = 'r' )
  plt.xlabel('R (arcsec)')
  plt.ylabel(r'$V_{circ}$ (km/s)')
  #plt.title('')
  plt.savefig("figures/NGC"+gal+"/Vcirc_JAM_MC_NGC"+gal+".pdf")
  plt.close()

  #... save Vcirc in a file
  np.savez('data_output/NGC'+gal+'/Vcirc_NGC'+gal, rad=rad, vcirc_med=vcirc_med, vcirc_up=vcirc_up, vcirc_dn=vcirc_dn)

  data2 = np.load('data_output/NGC'+gal+'/Vcirc_NGC'+gal+'.npz')
  rad=data2['rad']
  vcirc_med=data2['vcirc_med']
  vcirc_up=data2['vcirc_up']
  vcirc_dn=data2['vcirc_dn']



 
