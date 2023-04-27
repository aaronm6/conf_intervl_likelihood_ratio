import sys, os
import numpy as np
import scipy.stats as st
import profile_likelihood as plr

#s_true_i_dflt = np.logspace(np.log10(0.25), np.log10(20),24) - 0.25
#s_true_i_dflt = np.r_[0.,np.logspace(-3,np.log10(20),23)]
s_true_i_dflt = np.r_[np.logspace(np.log10(0.25),np.log10(20.25),24) - 0.25,0.001]
s_true_i_dflt.sort()
#s_true_i_dflt = s_true_i_dflt[:-1]

_f_sig_default = st.norm(loc=0., scale=1.)
_g_bg_default = st.norm(loc=3., scale=1.)

def gen_toys_and_fit(
    b_true=12., 
    s_true_i='default', 
    f_sig_dist='default',
    g_bg_dist='default',
    nsim=int(1e4)):
    
    if isinstance(s_true_i,str) and (s_true_i == 'default'):
        s_true_i = s_true_i_dflt
    if not isinstance(s_true_i, np.ndarray):
        raise TypeError("Input 's_true_i' must be a numpy array of numerical dtype, or a string 'default'")
    try:
        s_true_i = s_true_i.astype(float)
    except ValueError:
        print("s_true_i must have a numerical dtype")
    if s_true_i[0] != 0.:
        raise ValueError("First element of s_true_i must be 0.")
    
    if isinstance(f_sig_dist,str) and (f_sig_dist in ('default','Default','DEFAULT')):
        f_sig = _f_sig_default
    elif hasattr(f_sig_dist,'rvs') and (hasattr(f_sig_dist,'pdf') or hasattr(f_sig_dist,'pmf')):
            f_sig = f_sig_dist
    else:
        raise TypeError("Input 'f_sig_dist' must be 'default' or a frozen distribution")
    if isinstance(g_bg_dist,str) and (g_bg_dist in ('default','Default','DEFAULT')):
        g_bg = _g_bg_default
    elif hasattr(g_bg_dist,'rvs') and (hasattr(g_bg_dist,'pdf') or hasattr(g_bg_dist,'pmf')):
            g_bg = g_bg_dist
    else:
        raise TypeError("Input 'g_bg_dist' must be 'default' or a frozen distribution")
    
    xbD = {} # dict to hold the bg data from toys of all poi
    xsD = {} # dict to hold the signal data from toys of all poi
    mu_b = b_true
    Nb = np.empty((len(s_true_i),nsim),dtype=int)
    Ns = np.empty((len(s_true_i),nsim),dtype=int)
    
    mu_hat_i = np.empty((len(s_true_i), nsim), dtype=float)
    mu_hat1_i = np.empty((len(s_true_i), nsim), dtype=float)
    mu_hat1_0_i = np.empty((len(s_true_i), nsim), dtype=float)
    
    for kpoi, s_true in enumerate(s_true_i):
        Nb[kpoi,:] = st.poisson.rvs(b_true, size=nsim)
        Ns[kpoi,:] = st.poisson.rvs(s_true, size=nsim)
        xbD[f'xb_{kpoi}'] = g_bg.rvs(size=Nb[kpoi,:].sum())
        xsD[f'xs_{kpoi}'] = f_sig.rvs(size=Ns[kpoi,:].sum())
        f_ib = f_sig.pdf(xbD[f'xb_{kpoi}'])
        f_is = f_sig.pdf(xsD[f'xs_{kpoi}'])
        g_ib = g_bg.pdf(xbD[f'xb_{kpoi}'])
        g_is = g_bg.pdf(xsD[f'xs_{kpoi}'])
        mu_hat_i[kpoi,:] = plr.fit_mu(Nb[kpoi,:],Ns[kpoi,:],f_is,f_ib,g_is,g_ib,b_true)
        mu_hat1_i[kpoi,:] = np.vstack([mu_hat_i[kpoi,:],np.full(nsim,s_true)]).min(0)
        mu_hat1_0_i[kpoi,:] = np.vstack([mu_hat_i[0,:],np.full(nsim,s_true)]).min(0)
    
    outD = {}
    outD['b_true'] = np.r_[b_true] # True background expectation value
    outD['poi'] = s_true_i         # vector of signal expectation values (must start at zero)
    outD['Ns'] = Ns                # number of signal events for all toys for all poi
    outD['Nb'] = Nb                # number of observed BG events for all toys for all poi
    outD['mu_hat'] = mu_hat_i      # best fit signal expectation values when nu-test and nu-true are same
    outD['mu_hat1'] = mu_hat1_i    # best fit signal expectation values when nu-test and nu-true are same
    #                                1-sided, i.e. mu_hat1 = min(mu_hat, mu)
    outD['mu_hat1_0'] = mu_hat1_0_i
    outD.update(xbD)               # bg data along observable x, corresponding to each Nb_o
    outD.update(xsD)               # signal data along observable x, corresponding to each Ns_o
    return outD

def sim_loglikes(d=None,save_data=True,**kwargs):
    if ('f_sig_dist' not in kwargs) or \
    (isinstance(kwargs['f_sig_dist'],str) and kwargs['f_sig_dist']=='default'):
        f_sig = _f_sig_default
    else:
        f_sig = kwargs['f_sig_dist']
    if ('g_bg_dist' not in kwargs) or \
    (isinstance(kwargs['g_bg_dist'],str) and kwargs['g_bg_dist']=='default'):
        g_bg = _g_bg_default
    else:
        g_bg = kwargs['g_bg_dist']
    
    flag_d_input = True
    if d is None:
        flag_d_input = False
        d = gen_toys_and_fit(**kwargs)
        '''
        d = gen_toys_and_fit(
            b_true=b_true,
            s_true_i=s_true_i,
            f_sig_dist=f_sig,
            g_bg_dist=g_bg,
            xmean_sig=xmean_sig,
            xmean_bg=xmean_bg,
            nsim=nsim)
        '''
    nPOI, nsim = d['Ns'].shape
    b_true = d['b_true'][0]
    # For each poi (value of mu) need:      Called in code:
    #   L(mu | mu_true = mu)                L_mu_mu
    #   L(mu | mu_true = 0)                 L_mu_0
    #    L(mu_hat | mu_true = mu)           L_muhat_mu
    # L(mu_hat | mu_true = mu), mu_hat<=mu  L_muhat_mu1  (1 for 1-sided)
    #    L(mu_hat | mu_true = 0)            L_muhat_0 <--- same as L_muhat_mu (first row)
    #    L(0 | mu_true = 0)                 L_0_0   <--- same as L_0_mu (first row)
    #    L(0 | mu_true = mu)                L_0_mu
    L_mu_mu = np.empty((nPOI,nsim))
    L_mu_0 = np.empty((nPOI,nsim))
    L_muhat_mu = np.empty((nPOI,nsim))
    L_muhat1_mu = np.empty((nPOI,nsim))
    L_muhat1_0_0 = np.empty((nPOI,nsim))
    L_0_mu = np.empty((nPOI,nsim))
    earr = np.array([],dtype=float) # "e" for empty, "arr" for array
    f_ib0 = f_sig.pdf(d['xb_0'])
    g_ib0 = g_bg.pdf(d['xb_0'])
    for kpoi, poi in enumerate(d['poi']):
        #print(f'sim_loglikes: {kpoi = }/{len(d["poi"])}, {poi = }', flush=True)
        #with tictoc(print_on_exit=False) as tt:
        f_is = f_sig.pdf(d[f'xs_{kpoi}'])
        f_ib = f_sig.pdf(d[f'xb_{kpoi}'])
        g_is = g_bg.pdf(d[f'xs_{kpoi}'])
        g_ib = g_bg.pdf(d[f'xb_{kpoi}'])
        #print(f"PDF evaluation took {tt:0.2e} s")
        #del tt
        #with tictoc(print_on_exit=False) as tt:
        L_mu_mu[kpoi,:] = plr.log_L(poi,b_true,d['Ns'][kpoi,:],d['Nb'][kpoi,:],f_is,f_ib,g_is,g_ib)
        L_mu_0[kpoi,:] = plr.log_L(poi,b_true,d['Ns'][0,:],d['Nb'][0,:],earr,f_ib0,earr,g_ib0)
        L_muhat_mu[kpoi,:] = plr.log_Lhat(d['mu_hat'][kpoi,:],b_true,d['Ns'][kpoi,:],d['Nb'][kpoi,:],
            f_is, f_ib, g_is, g_ib)
        L_muhat1_mu[kpoi,:] = plr.log_Lhat(d['mu_hat1'][kpoi,:],b_true,d['Ns'][kpoi,:],d['Nb'][kpoi,:],
            f_is, f_ib, g_is, g_ib)
        L_muhat1_0_0[kpoi,:] = plr.log_Lhat(d['mu_hat1_0'][kpoi,:],b_true,d['Ns'][0,:],d['Nb'][0,:],
            earr, f_ib0, earr, g_ib0)
        L_0_mu[kpoi,:] = plr.log_L(0.,b_true,d['Ns'][kpoi,:],d['Nb'][kpoi,:],f_is,f_ib,g_is,g_ib)
        #print(f"Likelihood evaluation took {tt:0.2e} s")
    if not save_data:
        dkeys = list(d)
        for key in dkeys:
            if key.startswith('x'):
                del d[key]
    d['L_mu_mu']     = L_mu_mu
    d['L_mu_0']      = L_mu_0
    d['L_muhat_mu']  = L_muhat_mu
    d['L_muhat1_mu'] = L_muhat1_mu
    d['L_muhat1_0_0'] = L_muhat1_0_0
    d['L_0_mu']      = L_0_mu
    if not flag_d_input:
        return d

def sim_test_statistics(d=None, **kwargs):
    flag_d_input = True
    if d is None:
        d = sim_loglikes(**kwargs)
        flag_d_input = False
    nPOI, nsim = d['Ns'].shape
    # t-tilde-mu: -2*[logL(mu) - logL(mu_hat)], mu_hat >= 0
    #   evaluate for data(mu_true=mu) and data(mu_true=0)
    t_tilde_mu  = np.empty((nPOI,nsim))
    t_tilde_mu0 = np.empty((nPOI,nsim))
    # q_tilde-mu: -2*[logL(mu) - logL(mu_hat)], mu_hat >= 0, mu_hat<=mu
    #   evaluate for data(mu_true=mu) and data(mu_true=0)
    q_tilde_mu = np.empty((nPOI,nsim))
    q_tilde_mu0 = np.empty((nPOI,nsim))
    # q0: -2*[logL(0) - logL(mu_hat)], mu_hat >= 0
    #   evaluate for all mu, once
    q0 = np.empty((nPOI,nsim))
    # lep, or Q: -2*[logL(mu) - logL(0)]   <--- used in CLs original literature
    #   evaluate for data(mu_true=mu) and data(mu_true=0)
    # for each: need distribution when mu_true = mu, and when mu_true = 0
    lep = np.empty((nPOI,nsim))
    lep0 = np.empty((nPOI,nsim))
    # log_L(mu_s, mu_b, Ns, Nb, f_is, f_ib, g_is, g_ib)
    # log_Lhat(mu_s, mu_b, Ns, Nb, f_is, f_ib, g_is, g_ib)
    # Need: logL(mu|data=mu), logL(mu|data=0), logL(mu_hat|data=mu)
    #f_ib0 = f_sig.pdf(d['xb_0'])
    #g_ib0 = g_bg.pdf(d['xb_0'])
    earr = np.array([],dtype=np.float64) # empty array (different from np.empty)
    d['t_tilde_mu'] = -2 * (d['L_mu_mu'] - d['L_muhat_mu'])
    d['t_tilde_mu0'] = -2 * (d['L_mu_0'] - np.tile(d['L_muhat_mu'][0,:],(nPOI,1)))
    d['q_tilde_mu'] = -2 * (d['L_mu_mu'] - d['L_muhat1_mu'])
    d['q_tilde_mu0'] = -2 * (d['L_mu_0'] - d['L_muhat1_0_0'])
    d['q0'] = -2 * (d['L_0_mu'] - d['L_muhat_mu'])
    d['lep'] = -2 * (d['L_0_mu'] - d['L_mu_mu'])
    d['lep0'] = -2 * (np.tile(d['L_mu_mu'][0,:],(nPOI,1)) - d['L_mu_0'])
    
    if not flag_d_input:
        return d

def p_val_prep(a, b, direction='right'):
    """
    This uses plr.p_vals_ordered, but prepares the arrays for that function (i.e.
    it sorts them) and then prepares its output (i.e. unsorts that).
    
    direction='right' indicates the p-value should be calculated by integrating to the right
    direction='left'  indicates...  to the left.  
    """
    if not isinstance(a,np.ndarray) or (a.dtype != np.float64) or (a.ndim != 1):
        raise TypeError("Input 'a' must be a 1-D numpy array of dtype float64")
    if not isinstance(b,np.ndarray) or (b.dtype != np.float64) or (b.ndim != 1):
        raise TypeError("Input 'b' must be a 1-D numpy array of dtype float64")
    direction = direction.lower()
    if direction not in ('left','right'):
        raise ValueError("'direction' must be either 'left' or 'right'")
    a_as = a.argsort()
    b_as = b.argsort()
    pvs_raw = plr.p_vals_ordered(a[a_as], b[b_as], direction=direction)
    pvals_out = pvs_raw[b_as.argsort()]
    return pvals_out

def p_values(d=None, **kwargs):
    flag_d_input = True
    if d is None:
        d = sim_test_statistics(**kwargs)
        flag_d_input = False
    nPOI, nsim = d['Ns'].shape
    
    pvals0_t_tilde_mu = np.empty((nPOI,nsim))
    pval_0evts_t_tilde_mu = np.empty((nPOI,2))
    CLs_t_tilde_mu = np.empty((nPOI,nsim))
    
    pvals0_q_tilde_mu = np.empty((nPOI,nsim))
    CLs_q_tilde_mu = np.empty((nPOI,nsim))
    
    pvals0_q0 = np.empty((nPOI,nsim))
    CLs_q0    = np.empty((nPOI,nsim))
    
    pvals0_LEP = np.empty((nPOI,nsim))
    CLs_LEP    = np.empty((nPOI,nsim))
    
    for kpoi, poi in enumerate(d['poi']):
        pvals0_t_tilde_mu[kpoi,:] = p_val_prep(d['t_tilde_mu'][kpoi,:],d['t_tilde_mu0'][kpoi,:],
            direction='right')
        pval_0evts_t_tilde_mu[kpoi,:] = (d['t_tilde_mu'][kpoi,:] >= (2*poi)).sum() / d['t_tilde_mu'].shape[1]
        pvals00 = p_val_prep(d['t_tilde_mu0'][kpoi,:],d['t_tilde_mu0'][kpoi,:],direction='right')
        cut_CLb0 = pvals00 != 0.
        CLs_t_tilde_mu[kpoi,cut_CLb0] = pvals0_t_tilde_mu[kpoi,cut_CLb0] / pvals00[cut_CLb0]
        CLs_t_tilde_mu[kpoi,~cut_CLb0] = 20.
        
        pvals0_q_tilde_mu[kpoi,:] = p_val_prep(d['q_tilde_mu'][kpoi,:],d['q_tilde_mu0'][kpoi,:],
            direction='right')
        pvals00 = p_val_prep(d['q_tilde_mu0'][kpoi,:],d['q_tilde_mu0'][kpoi,:],direction='right')
        cut_CLb0 = pvals00 != 0.
        CLs_q_tilde_mu[kpoi,cut_CLb0] = pvals0_q_tilde_mu[kpoi,cut_CLb0] / pvals00[cut_CLb0]
        CLs_q_tilde_mu[kpoi,~cut_CLb0] = 20.
        
        pvals0_q0[kpoi,:] = p_val_prep(d['q0'][kpoi,:],d['q0'][0,:],direction='left')
        pvals00 = p_val_prep(d['q0'][0,:],d['q0'][0,:],direction='left')
        cut_CLb0 = pvals00 != 0.
        CLs_q0[kpoi,cut_CLb0] = pvals0_q0[kpoi,cut_CLb0] / pvals00[cut_CLb0]
        CLs_q0[kpoi,~cut_CLb0] = 20.
        
        kw_lep = {'nan_policy':'omit', 'kind':'weak'}
        pvals0_LEP[kpoi,:] = p_val_prep(d['lep'][kpoi,:],d['lep0'][kpoi,:],direction='left')
        pvals00 = p_val_prep(d['lep0'][kpoi,:],d['lep0'][kpoi,:], direction='left')
        cut_CLb0 = pvals00 != 0.
        CLs_LEP[kpoi,cut_CLb0] = pvals0_LEP[kpoi,cut_CLb0] / pvals00[cut_CLb0]
        CLs_LEP[kpoi,~cut_CLb0] = 20.
    outD = {}
    outD['pval0_t_tilde_mu'] = pvals0_t_tilde_mu
    outD['pval_0evts_t_tilde_mu'] = pval_0evts_t_tilde_mu
    outD['CLs_t_tilde_mu']   = CLs_t_tilde_mu
    outD['pval0_q_tilde_mu'] = pvals0_q_tilde_mu
    outD['CLs_q_tilde_mu']   = CLs_q_tilde_mu
    outD['pval0_q0']         = pvals0_q0
    outD['CLs_q0']           = CLs_q0
    outD['pval0_lep']        = pvals0_LEP
    outD['CLs_LEP']          = CLs_LEP
    d.update(outD)
    if not flag_d_input:
        return d

_q_n2sig = st.norm.cdf(-2)
_q_n1sig = st.norm.cdf(-1)
_q_median = 0.5
_q_p1sig = st.norm.cdf(1)
_q_p2sig = st.norm.cdf(2)

def get_UL_quantiles(d, key='pval0_t_tilde_mu', p0=0.1):
    b_true = d['b_true'][0]
    LL, UL, maxPoi = plr.get_limits(d['poi'], d[key], p0=p0)
    UL_n2sig = np.quantile(UL,_q_n2sig)
    UL_n1sig = np.quantile(UL,_q_n1sig)
    UL_median = np.quantile(UL,_q_median)
    UL_p1sig = np.quantile(UL,_q_p1sig)
    UL_p2sig = np.quantile(UL,_q_p2sig)
    ULs = np.r_[UL_n2sig, UL_n1sig, UL_median, UL_p1sig, UL_p2sig]
    return ULs










