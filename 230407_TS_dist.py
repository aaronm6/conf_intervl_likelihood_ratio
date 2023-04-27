import profile_likelihood as plr
import likelihood_1d_unbinned as LL

'''
assumed: 
from numpy import *, import scipy.stats as st, from matplotlib.pyplot import *, from aLib import *
'''

ntoys = int(1e5)
def gen_struct(b=0.5,mu=0.9,nsim=ntoys):
    d = LL.gen_toys_and_fit(b_true=b,s_true_i=r_[0.,mu],nsim=nsim)
    LL.sim_loglikes(d=d,save_data=True)
    LL.sim_test_statistics(d=d)
    return d

TS = 't_tilde_mu' # other options available

b, mu = .05, 2.0
d = gen_struct(b=b,mu=mu)

flag_inset = False

x_ts = linspace(0,10,200)
dx = x_ts[1] - x_ts[0]
xbnds = [0,8]

# x vector of histograms
x_ts -= 0.5 * dx

# histogram for 0, 1, 2, 3 observed signal events (specifically signal, not total)
n_ts_full = histogram(d[TS][1,:],x_ts)[0]
Ns_max = 6
n_ts = empty((Ns_max+1,len(x_ts)-1))

for k in range(Ns_max+1):
    cutT = d['Ns'][1,:] == k
    n_ts[k,:] = histogram(d[TS][1,cutT],x_ts)[0]

#figure(51, figsize=(6.4,9)); clf()
figure(51, figsize=(8,9)); clf()
ax1 = subplot(2,1,1)
ax2 = subplot(2,1,2)
squeeze_subplots()

#subplot(2,1,2)
sca(ax2)
lstairs(x_ts, n_ts_full/ntoys/dx,'-',lw=4,color=r_[1,1,1]*.1)

cyclerstep(3 - Ns_max)
#cyclerstep(-1)
#for k in range(3,-1,-1):
for k in range(Ns_max,-1,-1):
    #print(f"{k = }, {cyclerstep(0)}")
    lstairs(x_ts,n_ts[k,:]/ntoys/dx,'-',lw=1.5)


'''
leg([0,5,4,3,2,1],
    f"All toys ($\mu={mu:0.2f}$)",
    "Toys with $N_\mathrm{s}=0$",
    "Toys with $N_\mathrm{s}=1$",
    "Toys with $N_\mathrm{s}=2$",
    "Toys with $N_\mathrm{s}=3$",
    "Toys with $N_\mathrm{s}=4$",
    fontsize=12)
'''
leg([0] + list(range(Ns_max+1,0,-1)),
    f"All toys ($\mu={mu:0.2f}$)",
    *[f"Toys with $N_\mathrm{{s}}={II}$" for II in range(Ns_max+1)],
    fontsize=12)

xlim(xbnds)
xlabel(f'$\\tilde{{t}}_\mu=-2[\log\!L(\mu={mu:0.2f}) - \log\!L(\mu=\hat{{\mu}})]$')
ylabel("Probability Density")
ylim([1.8e-3,20])
tickxm(1)
text(xbnds[0]+.15,ylim()[1]*.7,
    '$$\log\!L(\mu)=-\mu-b+\sum_i\log\left[\mu f_\mathrm{sig}(x_i)+b f_\mathrm{bg}(x_i)\\right]$$',
    ha='left',va='top',fontsize=13, backgroundcolor='w')

#subplot(2,1,1)
sca(ax1)
x_L = linspace(*xbnds,int(1e4))
plot(x_L, st.percentileofscore(d[TS][1,:],x_L,kind='weak')/100,'-',lw=2.5)
ylabel('Cumulative Probability')
xlim(xbnds)
ylim([0,1])
cyclerstep(-1)
plot(xlim(),r_[1,1]*.9,'--',lw=1)
tickxm(1)
tickym(.1)
#grid()
title(f"$\mu = {mu:0.2f},\quad b = {b:0.2f}$")

tc = st.scoreatpercentile(d[TS][1,:],90.)
ax1.plot(r_[1,1]*tc,ax1.get_ylim(),'k--',lw=1)
ax2.plot(r_[1,1]*tc,ax2.get_ylim(),'k--',lw=1)


f_sig = st.norm(loc=0., scale=1.)
g_bg = st.norm(loc=3., scale=1.)
xi_L = linspace(-4,8,500)
if flag_inset:
    with rc_context({'font.size':12.0}):
        ax11 = ax1.inset_axes([.5,.15,.48,.65])
        ax11.plot(xi_L, b*g_bg.pdf(xi_L), 'k-',lw=3.)
        cyclerstep(1, ax=ax11)
        ax11.plot(xi_L, mu*f_sig.pdf(xi_L),'-',lw=3.)
        ax11.set_xlim([-4,8])
        tickxm(2,ax=ax11)
        ax11.set_ylim([0,3])
        #ax11.set_ylim([0,ax11.get_ylim()[1]])
        #setp(ax11.get_xticklabels(), backgroundcolor='w')
        #setp(ax11.get_yticklabels(), backgroundcolor='w')
        ax11.set_xlabel('$x$',labelpad=-1.)#,backgroundcolor='w')
        ax11.set_ylabel('Number density along $x$')#,labelpad=5.,backgroundcolor='w')
        ax11.set_title('Example toy experiment with $N_\mathrm{s}=0$',fontsize=12)#,backgroundcolor='w',pad=8.)
        #leg([1, 0], 'Signal density','Background density',ax=ax11, fontsize=10.)
        #leg([1, 0], '$\mu f_\mathrm{sig}(x)$','$b f_\mathrm{bg}(x)$',ax=ax11, fontsize=10.)
        cyclerstep(-1, ax=ax11)
        N_targ = 0
        i_1stNt = nonzero(d['Ns'][1,:] == N_targ)[0].min()
        Ncmsm = r_[0,d['Ns'][1,:]].cumsum()
        x_i = d['xs_1'][Ncmsm[i_1stNt]:(Ncmsm[i_1stNt]+N_targ)]
        ax11.plot(vstack([x_i,x_i,full(N_targ,nan)]).flatten('F'),
            tile([0,.2,nan],N_targ),'-',lw=1)
        #ADD BG EVENT PLOTTING
        Nbg = d['Nb'][1,i_1stNt]
        Nb_cmsm = r_[0,d['Nb'][1,:]].cumsum()
        xb_i = d['xb_1'][Nb_cmsm[i_1stNt]:(Nb_cmsm[i_1stNt]+Nbg)]
        ax11.plot(vstack([xb_i,xb_i,full(Nbg,nan)]).flatten('F'),
            tile([0,.5,nan],Nbg),'k-',lw=.75)
        leg([1, 0,3], '$\mu f_\mathrm{sig}(x)$','$b f_\mathrm{bg}(x)$','BG data point',
            ax=ax11, fontsize=10.)
        
    TS0 = d[TS][1,i_1stNt]
    ax1.plot(TS0,st.percentileofscore(d[TS][1,:],TS0,kind='strict')/100,'ro',markerfacecolor='none')
    cut_N0 = d['Ns'][1,:] == 0
    pvals_N0 = 1-st.percentileofscore(d[TS][1,:],d[TS][1,cut_N0],kind='strict')/100
    p_val_med = median(pvals_N0)
    ax1.text(.25,.2,f"Median $p$-value for\nBG-only experiments: {p_val_med:0.3f}",va='top')

