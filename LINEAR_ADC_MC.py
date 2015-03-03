import emcee
import math
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.stats as ss
import scipy.interpolate as interp
from astropy.table import Table
from readcol import readcol
import triangle


def log_adc(p, R, vobs, sobs, evobs, esobs,kv_in,incl_in, beta_in,sig0_in,ksig_in):

    kv, incl, beta, sig0, ksig = p[0], p[1], p[2], p[3], p[4] 
    
    ar = np.ones(len(R))
    sigr = sig0+ksig*R

    vmod = (kv*R)*np.sin(math.pi/180*incl)
    smod = np.sqrt(1 - beta*np.cos(math.pi/180*incl)**2 + 0.5*(ar-1)*np.sin(math.pi/180*incl)**2)*sigr
    
    
    print 'ideg=', incl        
    priors = ss.uniform.logpdf(kv,loc=0,scale=10)+\
             ss.uniform.logpdf(incl,loc=0,scale=90)+\
             ss.uniform.logpdf(beta,loc=-1,scale=2)+\
             ss.uniform.logpdf(sig0,loc=0,scale=300)+\
             ss.uniform.logpdf(ksig,loc=-5,scale=10)+\
             ss.norm.logpdf(kv,loc=kv_in,scale=0.5)+\
             ss.norm.logpdf(incl,loc=incl_in,scale=5)+\
             ss.norm.logpdf(beta,loc=beta_in,scale=0.5)+\
             ss.norm.logpdf(sig0,loc=sig0_in,scale=1)+\
             ss.norm.logpdf(ksig,loc=ksig_in,scale=0.5)
    

    if np.isfinite(priors) == False:
        return -np.inf

    p1 = (vmod-vobs)**2/evobs**2
    p1 = np.nansum(p1)
        
    p2 = (smod-sobs)**2/esobs**2
    p2 = np.nansum(p2)
                
    lp = - p1 - p2 + priors

    if np.isnan(lp):
        return -np.inf

    return lp



# %&%&%&%&%&%&%&%&%&%&%&
# Reading all the files
# %&%&%&%&%&%&%&%&%&%&%&

gals=['628','1042','3346','4487','4775','5585','5668']

kv_all= [1.08,1.50,4.51,3.13,1.44,0.42,2.11]
incl_all=[24,39,29,47,21,50,24]
beta_all=[0.40,0.25,0.25,0.25,0.15,0.15,0.15]
#beta_all=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
sig0_all=[59.01,53.90,57.46,60.80,58.65,49.68,53.36]
ksig_all=[1.40,0.08,0.20,-0.19,-0.13,-0.18,0.50]

for j in range(len(gals)):
    gal=gals[j]
    print 'gal=',gal
    
    kv_in=kv_all[j]
    incl_in=incl_all[j]
    beta_in=beta_all[j]
    sig0_in=sig0_all[j]
    ksig_in=ksig_all[j]

    sfilename = 'data_input/Sigma/Fourier_Sbin_'+gal+'.txt'
    vfilename = 'data_input/Velocity/Fourier_Vbin_'+gal+'.txt'
    mgefilename = 'mge/mge_NGC'+gal+'.txt'



    R, _, _, _, _, vobs, evobs, _, _, _ = readcol(vfilename, skipline = 5, twod=False)
    _,_,_,_,_,sobs,esobs,_,_ = readcol(sfilename, skipline = 5, twod=False)
    _, I0obs, spobs, _, _ = readcol(mgefilename, twod=False, skipline = 2)
    #R, Iobs = readcol(mufilename, skipline = 2, twod=False)
    esobs=esobs+2
    evobs=evobs+2



    # %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    # MCMC for Vobs and sigma_obs
    # %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    print 'walkers'
    # Set the walkers
    ndim, nwalkers = 5,60
    p0 = np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.randn(nwalkers)*0.5+kv_in
    p0[:,1] = np.random.randn(nwalkers)*5+incl_in
    p0[:,2] = np.random.randn(nwalkers)*0.5+beta_in
    p0[:,3] = np.random.randn(nwalkers)*1+sig0_in
    p0[:,4] = np.random.randn(nwalkers)*0.5+ksig_in

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_adc,
            args=[R,vobs,sobs,evobs,esobs,kv_in,incl_in, beta_in,sig0_in,ksig_in], threads=12)

    pos, prob, state = sampler.run_mcmc(p0, 400)
    
    
      # Let's look at some of the sample chains.  Feel free to try different chains.  The order of elements is `sampler.chain[chain,step,parameter]`

     # In[11]:
    print 'Chain figure'
    fig = plt.figure(figsize=(10,6))
    plt.subplot(3,2,1)
    plt.title(r'NGC'+gal)
    plt.plot(sampler.chain[:,:,0].T)
    plt.ylabel(r'Chain for $k_v$')
    plt.subplot(3,2,2)
    plt.plot(sampler.chain[:,:,1].T)
    plt.ylabel(r'Chain for $ideg$')
    plt.subplot(3,2,3)
    plt.plot(sampler.chain[:,:,2].T)
    plt.ylabel(r'Chain for $\beta_z$')
    plt.subplot(3,2,4)
    plt.plot(sampler.chain[:,:,3].T)
    plt.ylabel(r'Chain for $\sigma_0$')
    plt.subplot(3,2,5)
    plt.plot(sampler.chain[:,:,4].T)
    plt.ylabel(r'Chain for $k_{\sigma}$')
   
    #---------------------------------------------------
    plt.tight_layout() # This tightens up the spacing
    plt.savefig("figures/chains/NGC"+gal+"_ADC_lin_chain.pdf")
    plt.close()
    
    
    
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos,1000)
    
    #------------- triangle -------
    print 'Triangle figure'
    fig = plt.figure(figsize=(10,6))
    triangle.corner(sampler.flatchain, labels=["$k_v$","$ideg$", "$beta_z$", "$\sigma_0$", "$k_{\sigma}$"])
    #fig.gca().annotate("NGC"+gal, xy=(0.5, 1.0), xycoords="figure fraction",xytext=(0, -5), textcoords="offset points",ha="center", va="top")
    plt.savefig("figures/triangles/NGC"+gal+"_ADC_lin_triangle.pdf")
    plt.close()
    
   

    #------------------------
    bins_adc = 50

    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3,1)
    patches = plt.hist(sampler.flatchain[:,0],bins=bins_adc)
    plt.title(r'Distribution of $k_{v}$')
    plt.subplot(2,3,2)
    patches = plt.hist(sampler.flatchain[:,1],bins=bins_adc)
    plt.title(r'Distribution of incl')
    plt.subplot(2,3,3)
    patches = plt.hist(sampler.flatchain[:,2],bins=bins_adc)
    plt.title(r'Distribution of $\beta_{z}$')
    plt.subplot(2,3,4)
    patches = plt.hist(sampler.flatchain[:,3],bins=bins_adc)
    plt.title(r'Distribution of $\sigma_{0}$')
    plt.subplot(2,3,5)
    patches = plt.hist(sampler.flatchain[:,4],bins=bins_adc)
    plt.title(r'Distribution of $k_{\sigma}$')

    plt.savefig("figures/NGC"+gal+"/Vcirc_ADC_histo.pdf")
    plt.close()

    kv_dist = sampler.flatchain[:,0]
    incl_dist = sampler.flatchain[:,1]
    betaz_dist = sampler.flatchain[:,2]
    sig0_dist = sampler.flatchain[:,3]
    ksig_dist = sampler.flatchain[:,4]

    ar_dist=np.ones(sampler.flatchain.shape)

    kv_med = np.median(sampler.flatchain[:,0])
    incl_med = np.median(sampler.flatchain[:,1])
    betaz_med = np.median(sampler.flatchain[:,2])
    sig0_med = np.median(sampler.flatchain[:,3])
    ksig_med = np.median(sampler.flatchain[:,4])


    kv_plus = np.percentile(sampler.flatchain[:,0], 75)- np.median(sampler.flatchain[:,0])
    incl_plus = np.percentile(sampler.flatchain[:,1], 75)- np.median(sampler.flatchain[:,1])
    betaz_plus = np.percentile(sampler.flatchain[:,2], 75)- np.median(sampler.flatchain[:,2])
    sig0_plus = np.percentile(sampler.flatchain[:,3], 75)- np.median(sampler.flatchain[:,3])
    ksig_plus = np.percentile(sampler.flatchain[:,4], 75)- np.median(sampler.flatchain[:,4])


    kv_minus = np.median(sampler.flatchain[:,0]) - np.percentile(sampler.flatchain[:,0], 25)
    incl_minus = np.median(sampler.flatchain[:,1]) - np.percentile(sampler.flatchain[:,1], 25)
    betaz_minus = np.median(sampler.flatchain[:,2]) - np.percentile(sampler.flatchain[:,2], 25)
    sig0_minus = np.median(sampler.flatchain[:,3]) - np.percentile(sampler.flatchain[:,3], 25)
    ksig_minus = np.median(sampler.flatchain[:,4]) - np.percentile(sampler.flatchain[:,4], 25)

    #-----------------------------------------------
    # Array for the medians
    filename='data_output/Tables/Table_NGC_'+gal+'.txt'
    medians = [kv_med, incl_med,betaz_med,sig0_med,ksig_med]

    # Array for the upper percentiles
    ups = [kv_plus, incl_plus,betaz_plus,sig0_plus,ksig_plus]

    # Array for the lower percentiles
    lws = [kv_minus, incl_minus,betaz_minus,sig0_minus,ksig_minus]

    t = Table([medians, ups, lws], names = ('medians(kv,incl,betaz,sig0,ksig)','ups','lws'))
    t.write(filename, format = 'ascii')
    #-------------------------------------------------

    print '%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&'
    print 'Fitted parameters'
    print '%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&'

    print 'kv: ', np.median(sampler.flatchain[:,0]), \
        '+', np.percentile(sampler.flatchain[:,0], 75) - np.median(sampler.flatchain[:,0]),\
        '-', np.median(sampler.flatchain[:,0]) - np.percentile(sampler.flatchain[:,0], 25)
    print 'incl: ', np.median(sampler.flatchain[:,1]), \
        '+', np.percentile(sampler.flatchain[:,1], 75) - np.median(sampler.flatchain[:,1]),\
        '-', np.median(sampler.flatchain[:,1]) - np.percentile(sampler.flatchain[:,1], 25) 
    print 'betaz: ', np.median(sampler.flatchain[:,2]), \
        '+', np.percentile(sampler.flatchain[:,2], 75) - np.median(sampler.flatchain[:,2]),\
        '-', np.median(sampler.flatchain[:,2]) - np.percentile(sampler.flatchain[:,2], 25)
    print 'sig0: ', np.median(sampler.flatchain[:,3]), \
        '+', np.percentile(sampler.flatchain[:,3], 75) - np.median(sampler.flatchain[:,3]),\
        '-', np.median(sampler.flatchain[:,3]) - np.percentile(sampler.flatchain[:,3], 25)
    print 'ksig: ', np.median(sampler.flatchain[:,4]), \
        '+', np.percentile(sampler.flatchain[:,4], 75) - np.median(sampler.flatchain[:,4]),\
        '-', np.median(sampler.flatchain[:,4]) - np.percentile(sampler.flatchain[:,4], 25)             



    Vphi_mod = np.zeros([len(R),len(sampler.flatchain[:,0])])
    alphar_mod = np.ones([len(R),len(sampler.flatchain[:,0])])
    sigr_mod = np.zeros([len(R),len(sampler.flatchain[:,0])])
    dlnSigR2_dlnR_mod = np.zeros([len(R),len(sampler.flatchain[:,0])])


    Vphi_obs = np.zeros([len(R),len(sampler.flatchain[:,0])])
    sigr_obs = np.zeros([len(R),len(sampler.flatchain[:,0])])
    sobs_fin= np.zeros([len(R),len(sampler.flatchain[:,0])])
    vobs_fin= np.zeros([len(R),len(sampler.flatchain[:,0])])

    for j in range(len(R)):

        Vphi_mod[j,:] = kv_dist*R[j]
        sigr_mod[j,:] = sig0_dist + ksig_dist*R[j]
        dlnSigR2_dlnR_mod[j,:] = 2*ksig_dist*R[j]/(sig0_dist + ksig_dist*R[j]) 


    for j in range(len(R)):

        Vphi_obs[j,:] = vobs[j]/np.sin(math.pi/180.*incl_dist)
        sigr_obs[j,:] = sobs[j]/np.sqrt(1 - \
                                        betaz_dist*np.cos(math.pi/180*incl_dist)**2 ) #+ \
                                        #0.5*(ar_dist-1)*np.sin(math.pi/180*incl_dist)**2)
        sobs_fin[j,:] = (sig0_dist + ksig_dist*R[j])*np.sqrt(1 - \
                                        betaz_dist*np.cos(math.pi/180*incl_dist)**2) # + \
                                      # 0.5*(ar_dist-1)*np.sin(math.pi/180*incl_dist)**2)

        vobs_fin[j,:] = kv_dist*R[j]*np.sin(math.pi/180.*incl_dist)

        
                                        
    # Itot = I(R), dItot = sum(I0j*exp(-0.5R^2/spj^2)*(-R/spj))    
    Itot = np.zeros(len(R))
    dItot = np.zeros(len(R))
    for j in range(len(R)):
        for i in range(len(I0obs)):
            
            Itot[j] = Itot[j] + I0obs[i]*np.exp(-0.5*R[j]**2/(spobs[i]**2)) 
            dItot[j] = dItot[j] + (-R[j]/(spobs[i]**2))*I0obs[i]*np.exp(-0.5*R[j]**2/(spobs[i]**2))
            

    dlnI_dlnR = R*dItot/Itot 
    dlnI_dlnRs = np.tile(dlnI_dlnR,(Vphi_mod.shape[1],1))
    dlnI_dlnRs = dlnI_dlnRs.T
     #===========================
    #... save the distributions in a file
    np.savez('data_output/NGC'+gal+'/NGC'+gal+'_distributions', \
    kv_dist=kv_dist , incl_dist=incl_dist, \
    betaz_dist=betaz_dist, sig0_dist=sig0_dist,ksig_dist =ksig_dist)

    data = np.load('data_output/NGC'+gal+'/NGC'+gal+'_distributions.npz')

    kv_dist=data['kv_dist']
    incl_dist=data['incl_dist']
    betaz_dist=data['betaz_dist']
    sig0_dist=data['sig0_dist'] 
    ksig_dist =data['ksig_dist']
    #==========================

    # %&%&%&%&%&%&%&%&%
    #    Final ADC
    # %&%&%&%&%&%&%&%&%    

    # From model...
    Vc2_mod = Vphi_mod**2 + sigr_mod**2*(-dlnI_dlnRs - dlnSigR2_dlnR_mod - 0.5*(1-alphar_mod))
    Vc_mod = np.sqrt(Vc2_mod)     

    # From observation + model...
    Vc2_obmod = Vphi_obs**2 + sigr_obs**2*(-dlnI_dlnRs - dlnSigR2_dlnR_mod - 0.5*(1-alphar_mod))
    Vc_obmod = np.sqrt(Vc2_obmod)


    Vc2_oblit = Vphi_obs**2 + sigr_obs**2
    print 'Print figures'
    fig = plt.figure(figsize=(10,6))
    #plt.plot(R,np.median(Vc_obmod, axis = 1), 'b')
    eVctop0 = np.percentile(Vc_obmod, 75, axis = 1) - np.median(Vc_obmod, axis = 1)
    eVcbot0 = np.median(Vc_obmod, axis = 1) - np.percentile(Vc_obmod, 25, axis = 1)
    #plt.errorbar(R,np.median(Vc_obmod, axis = 1), yerr = (eVctop0, eVcbot0), color = 'b' )

    plt.plot(R,np.median(Vc_mod, axis = 1), 'ro')
    eVctop = np.percentile(Vc_mod, 75, axis = 1) - np.median(Vc_mod, axis = 1)
    eVcbot = np.median(Vc_mod, axis = 1) - np.percentile(Vc_mod, 25, axis = 1)
    plt.errorbar(R,np.median(Vc_mod, axis = 1), yerr = (eVctop, eVcbot), color = 'r' )


    plt.savefig("figures/NGC"+gal+"/Vcirc_ADC_MCMC.pdf")
    plt.close()


    #... Vobs profiles
    fig = plt.figure(figsize=(10,6))
    plt.plot(R,np.median(Vphi_obs, axis=1), 'bo')
    eVctop1 = np.percentile(Vphi_obs, 75, axis = 1) - np.median(Vphi_obs, axis = 1)
    eVcbot1 = np.median(Vphi_obs, axis = 1) - np.percentile(Vphi_obs, 25, axis = 1)
    plt.errorbar(R,np.median(Vphi_obs, axis = 1), yerr = (eVctop1, eVcbot1), color = 'b' )


    plt.plot(R,np.median(Vphi_mod,axis=1), 'r-')
    eVctop2 = np.percentile(Vphi_mod, 75, axis = 1) - np.median(Vphi_mod, axis = 1)
    eVcbot2 = np.median(Vphi_mod, axis = 1) - np.percentile(Vphi_mod, 25, axis = 1)
    plt.errorbar(R,np.median(Vphi_mod, axis = 1), yerr = (eVctop2, eVcbot2), color = 'r' )

    plt.plot(R,vobs, 'g-')
    plt.errorbar(R,vobs, yerr = (evobs, evobs), color = 'g' )

    plt.savefig("figures/NGC"+gal+"/Vphi_profiles.pdf")
    plt.close()

    #... Sigma_obs profiles
    fig = plt.figure(figsize=(10,6))
    plt.plot(R,np.median(sigr_obs, axis=1), 'o')
    eVctop3 = np.percentile(sigr_obs, 75, axis = 1) - np.median(sigr_obs, axis = 1)
    eVcbot3 = np.median(sigr_obs, axis = 1) - np.percentile(sigr_obs, 25, axis = 1)
    plt.errorbar(R,np.median(sigr_obs, axis = 1), yerr = (eVctop3, eVcbot3), color = 'b' )

    plt.plot(R,np.median(sigr_mod,axis=1), 'ro')
    eVctop4 = np.percentile(sigr_mod, 75, axis = 1) - np.median(sigr_mod, axis = 1)
    eVcbot4 = np.median(sigr_mod, axis = 1) - np.percentile(sigr_mod, 25, axis = 1)
    plt.errorbar(R,np.median(sigr_mod, axis = 1), yerr = (eVctop4, eVcbot4), color = 'r' )

    #plt.plot(R,sobs, 'g-')
    #plt.errorbar(R,sobs, yerr = (esobs, esobs), color = 'g' )

    plt.savefig("figures/NGC"+gal+"/SgR_profiles.pdf")
    plt.close()

    #... Sobs profiles
    fig = plt.figure(figsize=(10,6))
    plt.plot(R,sobs, 'go')
    plt.errorbar(R,sobs, yerr = (esobs, esobs), color = 'g' )

    plt.plot(R,np.median(sobs_fin,axis=1), 'ro')
    eVctop5 = np.percentile(sobs_fin, 75, axis = 1) - np.median(sobs_fin, axis = 1)
    eVcbot5 = np.median(sobs_fin, axis = 1) - np.percentile(sobs_fin, 25, axis = 1)
    plt.errorbar(R,np.median(sobs_fin, axis = 1), yerr = (eVctop5, eVcbot5), color = 'r' )

    plt.savefig("figures/NGC"+gal+"/Sobs_profiles.pdf")
    plt.close()

    #...Vobs_profile
    fig = plt.figure(figsize=(10,6))
    plt.plot(R,vobs, 'go')
    plt.errorbar(R,vobs, yerr = (esobs, esobs), color = 'g' )

    plt.plot(R,np.median(vobs_fin,axis=1), 'ro')
    eVctop6 = np.percentile(vobs_fin, 75, axis = 1) - np.median(vobs_fin, axis = 1)
    eVcbot6 = np.median(vobs_fin, axis = 1) - np.percentile(vobs_fin, 25, axis = 1)
    plt.errorbar(R,np.median(vobs_fin, axis = 1), yerr = (eVctop6, eVcbot6), color = 'r' )

    plt.savefig("figures/NGC"+gal+"/Vobs_profiles.pdf")
    plt.close()
########################################################################################
    print 'Save in files'
    #... save Vcirc in a file
    np.savez('data_output/NGC'+gal+'/Vcirc_NGC'+gal, \
    #...V and S observed values
    R=R, vobs=vobs, evobs=evobs, sobs=sobs, esobs=esobs,\
    #...V and S Modeled values
    vobs_fin=vobs_fin, eVctop6=eVctop6, eVcbot6=eVcbot6,\
    sobs_fin=sobs_fin, eVctop5=eVctop5, eVcbot5=eVcbot5,\
       #...Vph and SigR observed
    sigr_obs=sigr_obs, eVctop3=eVctop3, eVcbot3=eVcbot3,\
    Vphi_obs=Vphi_obs, eVctop1=eVctop1, eVcbot1=eVcbot1,\
    #...Vph and SigR modeled
    sigr_mod=sigr_mod, eVctop4=eVctop4, eVcbot4=eVcbot4,\
    Vphi_mod=Vphi_mod, eVctop2=eVctop2, eVcbot2=eVcbot2,\
    #...Vc_ADC observed
    Vc_obmod=Vc_obmod, eVctop0=eVctop0, eVcbot0=eVcbot0,\
    # Vc_ADC modeled
    Vc_mod=Vc_mod, eVctop=eVctop, eVcbot=eVcbot)

    #data2 = np.load('data_output/NGC'+gal+'/Vcirc_NGC'+gal+'.npz')
    #R=data2['R']
    #vobs=data2['vobs']
    #evobs=data2['evobs']
    #sobs=data2['sobs']
    #esobs=data2['esobs']
    #------------
    #vobs_fin=data2['vobs_fin']
    #eVctop6=data2['eVctop6']
    #eVcbot6=data2['eVcbot6']
    #sobs_fin=data2['sobs_fin']
    #eVctop5=data2['eVctop5']
    #eVcbot5=data2['eVcbot5']
    #---------------
    #sigr_obs=data2['sigr_obs']
    #eVctop3=data2['eVctop3']
    #eVcbot3=data2['eVcbot3']
    #Vphi_obs=data2['Vphi_obs']
    #eVctop1=data2['eVctop1']
    #eVcbot1=data2['eVcbot1']
    #---------------
    #sigr_mod=data2['sigr_mod']
    #eVctop4=data2['eVctop4']
    #eVcbot4=data2['eVcbot4']
    #Vphi_mod=data2['Vphi_mod']
    #eVctop2=data2['eVctop2']
    #eVcbot2=data2['eVcbot2']
    #-----------------
    #Vc_obmod=data2['Vc_obmod']
    #eVctop0=data2['eVctop0']
    #eVcbot0=data2['eVcbot0']
    #Vc_obmod=data2['Vc_obmod']
    #eVctop0=data2['eVctop']
    #eVcbot0=data2['eVcbot']
    #---------------
    print 'Here is the end!'

