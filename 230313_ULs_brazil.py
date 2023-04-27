import likelihood_1d_unbinned as toy
import profile_likelihood as plr

if 'ULs_ttm' not in locals():
    nsim = 10000
    #nsim = 30000
    
    b_range = logspace(-2,3,24)
    #b_range = b_range[5:]
    #b_range = b_range[:15]
    
    ULs_ttm = empty((5, len(b_range)))
    UL_ttm_0evt = empty(len(b_range))
    pval_0evt_t_tilde_mu = empty((len(b_range),25))
    ULs_qtm = empty((5, len(b_range)))
    ULs_q0 = empty((5, len(b_range)))
    ULs_lep = empty((5, len(b_range)))
    
    CULs_ttm = empty((5, len(b_range)))
    CULs_qtm = empty((5, len(b_range)))
    CULs_q0 = empty((5, len(b_range)))
    CULs_lep = empty((5, len(b_range)))
    b_added = []
    kb_added = []
    UL0_added = []
    for kb, b in enumerate(b_range):
        print(f'{kb = } / {len(b_range)-1}')
        d = toy.sim_test_statistics(b_true=b,nsim=nsim)
        print("  ...finished statistics sim")
        toy.p_values(d)
        print("  ...calculated p values")
        ULs_ttm[:,kb] = toy.get_UL_quantiles(d, key='pval0_t_tilde_mu')
        #UL_ttm_0evt[kb] = plr.get_limits(d['poi'],d['pval_0evts_t_tilde_mu'])[1].mean()
        pval_0evt_t_tilde_mu[kb,:] = d['pval_0evts_t_tilde_mu'][:,0]
        cutNb0 = d['Nb'][0,:] == 0
        if cutNb0.sum() > 0:
            kb_added.append(kb)
            b_added.append(b)
            _, t_ULs, _ = plr.get_limits(d['poi'],d['pval0_t_tilde_mu'])
            UL0_added.append(t_ULs[nonzero(cutNb0)[0].min()])
            UL_ttm_0evt[kb] = t_ULs[nonzero(cutNb0)[0].min()]
        #    LL, UL, maxPoi = plr.get_limits(d['poi'], d[key], p0=p0)
        print("     .. UL ttm done")
        ULs_qtm[:,kb] = toy.get_UL_quantiles(d, key='pval0_q_tilde_mu')
        print("     .. UL qtm done")
        ULs_q0[:,kb] = toy.get_UL_quantiles(d, key='pval0_q0')
        print("     .. UL q0 done")
        ULs_lep[:,kb] = toy.get_UL_quantiles(d, key='pval0_lep')
        print("     .. UL lep done")
        CULs_ttm[:,kb] = toy.get_UL_quantiles(d, key='CLs_t_tilde_mu')
        print("     .. CUL ttm done")
        CULs_qtm[:,kb] = toy.get_UL_quantiles(d, key='CLs_q_tilde_mu')
        print("     .. CUL qtm done")
        CULs_q0[:,kb] = toy.get_UL_quantiles(d, key='CLs_q0')
        print("     .. CUL q0 done")
        CULs_lep[:,kb] = toy.get_UL_quantiles(d, key='CLs_LEP')
        print("     .. CUL lep done")

# get 0-event ULs.... the method in plr.get_limits doesn't work right
# This assumes the POI vector is the same for all background values

for kb, b in enumerate(b_range):
    if kb not in kb_added:
        poi_val = d['poi'][-2]
        kp_p = len(d['poi']) - 1
        kp_m = len(d['poi']) - 2
        #kp_p = kp_m
        while pval_0evt_t_tilde_mu[kb,kp_m] < 0.1:
            #if d['poi'][kp_m] > poi_val:
            kp_p = kp_m
            poi_val = d['poi'][kp_m]
            kp_m -= 1
        #p_slope = diff(d['poi'][kp:(kp+2)]) / diff(pval_0evt_t_tilde_mu[kb,kp:(kp+2)])
        p_slope = diff(d['poi'][[kp_m,kp_p]]) / diff(pval_0evt_t_tilde_mu[kb,[kp_m,kp_p]])
        #UL_ttm_0evt[kb] = d['poi'][kp] + (0.1-pval_0evt_t_tilde_mu[kb,kp])*p_slope
        UL_ttm_0evt[kb] = d['poi'][kp_m] + (0.1-pval_0evt_t_tilde_mu[kb,kp_m])*p_slope

#------- PLOTS OF ULs WITH P-VALS
figure(71, figsize=(12.42, 8.11)); clf()

subplot(2,2,1) # t~mu
fill_between(b_range, ULs_ttm[0,:],ULs_ttm[4,:],color=r_[.98,.98,.55])
fill_between(b_range, ULs_ttm[1,:],ULs_ttm[3,:],color=r_[.5,.95,.5])
plot(b_range, ULs_ttm[2,:],'k--')
plot(b_range, UL_ttm_0evt,'r--')
#plot(b_added, UL0_added,':',color=r_[1,1,1]*.5)
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('$\\tilde{t}_\mu$ UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

subplot(2,2,2) # q~mu
fill_between(b_range, ULs_qtm[0,:],ULs_qtm[4,:],color=r_[.98,.98,.55])
fill_between(b_range, ULs_qtm[1,:],ULs_qtm[3,:],color=r_[.5,.95,.5])
plot(b_range, ULs_qtm[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('$\\tilde{q}_\mu$ UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

subplot(2,2,3) # q0
fill_between(b_range, ULs_q0[0,:],ULs_q0[4,:],color=r_[.98,.98,.55])
fill_between(b_range, ULs_q0[1,:],ULs_q0[3,:],color=r_[.5,.95,.5])
plot(b_range, ULs_q0[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('$q_0$ UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

subplot(2,2,4) # LEP
fill_between(b_range, ULs_lep[0,:],ULs_lep[4,:],color=r_[.98,.98,.55])
fill_between(b_range, ULs_lep[1,:],ULs_lep[3,:],color=r_[.5,.95,.5])
plot(b_range, ULs_lep[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('LEP UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

#------- PLOTS OF ULs WITH CLs
figure(72, figsize=(12.42, 8.11)); clf()

subplot(2,2,1) # t~mu
fill_between(b_range, CULs_ttm[0,:],CULs_ttm[4,:],color=r_[.98,.98,.55])
fill_between(b_range, CULs_ttm[1,:],CULs_ttm[3,:],color=r_[.5,.95,.5])
plot(b_range, CULs_ttm[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('$\\tilde{t}_\mu$ UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

subplot(2,2,2) # q~mu
fill_between(b_range, CULs_qtm[0,:],CULs_qtm[4,:],color=r_[.98,.98,.55])
fill_between(b_range, CULs_qtm[1,:],CULs_qtm[3,:],color=r_[.5,.95,.5])
plot(b_range, CULs_qtm[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('$\\tilde{q}_\mu$ UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

subplot(2,2,3) # q0
fill_between(b_range, CULs_q0[0,:],CULs_q0[4,:],color=r_[.98,.98,.55])
fill_between(b_range, CULs_q0[1,:],CULs_q0[3,:],color=r_[.5,.95,.5])
plot(b_range, CULs_q0[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('$q_0$ UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

subplot(2,2,4) # LEP
fill_between(b_range, CULs_lep[0,:],CULs_lep[4,:],color=r_[.98,.98,.55])
fill_between(b_range, CULs_lep[1,:],CULs_lep[3,:],color=r_[.5,.95,.5])
plot(b_range, CULs_lep[2,:],'k--')
xlabel('$\mu_\mathrm{bg}$ (Background expectation value)')
ylabel('LEP UL on $\mu_\mathrm{s}$ for background-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])

#figure(73, figsize=(9.2,6.5)); clf()
figure(73); clf()
fill_between(b_range, ULs_ttm[0,:],ULs_ttm[4,:],color=r_[.98,.98,.55])
fill_between(b_range, ULs_ttm[1,:],ULs_ttm[3,:],color=r_[.5,.95,.5])
plot(b_range, ULs_ttm[2,:],'k--')
plot(b_range, UL_ttm_0evt,':',color=r_[1,1,1,]*.5)
#plot(b_added, UL0_added,':',color=r_[1,1,1]*.5)
xlabel('$b$ (Background expectation value)')
#ylabel('$\\tilde{t}_\mu$ UL on $\mu_\mathrm{s}$ for background-only experiments')
ylabel('90\% upper limit on $\mu$ for BG-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
ylim([.3,30])
plot(xlim(),r_[1,1]*2.44,'r--',lw=1)
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])
leg([2,1,0,4,3],'Median sensitivity','1-$\sigma$ sensitivity band',
    '2-$\sigma$ sensitivity band','FC counting BG-free UL','UL if 0 events observed',
    fontsize=12)
legloc('upper left')
grid()

figure(74); clf()
fill_between(b_range, CULs_ttm[0,:],CULs_ttm[4,:],color=r_[.98,.98,.55])
fill_between(b_range, CULs_ttm[1,:],CULs_ttm[3,:],color=r_[.5,.95,.5])
plot(b_range, CULs_ttm[2,:],'k--')
xlabel('$b$ (Background expectation value)')
ylabel('90\% CLs UL on $\mu$ for BG-only experiments')
gca().set_xscale('log')
gca().set_yscale('log')
xlim((.01,b_range.max()))
plot(xlim(),r_[1,1]*2.44,'r--',lw=1)
ylim([.3,30])
gca().set_yticks([.5,1,2,5,10,20])
gca().set_yticklabels([f'{item:0.0f}' if item>=1 else f'{item}' for item in gca().get_yticks()])
leg([2,1,0,3],'Median sensitivity','1-$\sigma$ sensitivity band',
    '2-$\sigma$ sensitivity band','FC counting BG-free UL',
    fontsize=12)
legloc('upper left')




