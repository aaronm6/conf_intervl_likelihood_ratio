#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <math.h>

/* ----------------- <AUX FUNCTIONS> ----------------- */
double dlogL(
	double mus,
	double mub,
	npy_intp is0,
	npy_intp is1,
	npy_intp ib0,
	npy_intp ib1,
	double *fs_ii,
	double *gs_ii,
	double *fb_ii,
	double *gb_ii) {
	// This evaluates the derivative of the log-likelihood function for a single sim
	double xsum = 0.;
	for (npy_intp i=is0; i<is1; i++) {
		xsum += fs_ii[i] / (mus*fs_ii[i] + mub*gs_ii[i]);
	}
	for (npy_intp i=ib0; i<ib1; i++) {
		xsum += fb_ii[i] / (mus*fb_ii[i] + mub*gb_ii[i]);
	}
	double outval = xsum - 1.;
	return outval;
}

double f_logL(
	double mus,
	double mub,
	npy_intp is0,
	npy_intp is1,
	npy_intp ib0,
	npy_intp ib1,
	double *fs_ii,
	double *gs_ii,
	double *fb_ii,
	double *gb_ii) {
	// This evaluates the log-likelihood function for a single sim
	double xsum = 0.;
	for (npy_intp i=is0; i<is1; i++) {
		if ((mus*fs_ii[i] + mub*gs_ii[i]) > 0) {
			xsum += log(mus*fs_ii[i] + mub*gs_ii[i]);
		} else {
			xsum = -100000.;
		}
	}
	for (npy_intp i=ib0; i<ib1; i++) {
		if (mus*fb_ii[i] + mub*gb_ii[i] > 0) {
			xsum += log(mus*fb_ii[i] + mub*gb_ii[i]);
		} else {
			xsum = -200000.;
		}
		
	}
	double outval = xsum - mus - mub;
	return outval;
}
/* ----------------- </AUX FUNCTIONS> ----------------- */

/* ----------------- <MODULE FUNCTIONS> ----------------- */

static PyObject *meth_get_limits(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *nd_poi, *nd_pvals;
	double p0 = 0.1;
	static char *keywords[] = {"", "", "p0", NULL};
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O&O&|d", keywords,
		PyArray_Converter, &nd_poi,
		PyArray_Converter, &nd_pvals,
		&p0)) {
		return NULL;
	}
	// ------ input checking ------- //
	npy_intp nPOI = PyArray_SIZE(nd_poi);
	int poi_ndim = PyArray_NDIM(nd_poi);
	int poi_tp = PyArray_TYPE(nd_poi);
	if (poi_tp != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "poi must have dtype double/float64");
		return NULL;
	}
	if (poi_ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "poi must be a 1D array");
		return NULL;
	}
	
	int pval_ndim = PyArray_NDIM(nd_pvals);
	int pval_tp   = PyArray_TYPE(nd_pvals);
	npy_intp *pval_dims = PyArray_DIMS(nd_pvals);
	if (pval_tp != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "pval must have dtype double/float64");
		return NULL;
	}
	if (pval_ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "pval must be a 2D array");
		return NULL;
	}
	if (pval_dims[0] != nPOI) {
		PyErr_SetString(PyExc_TypeError, "pval 1st dimension must be same as length of poi");
		return NULL;
	}
	npy_intp nTOYS = pval_dims[1];
	npy_intp *pval_st_bytes = PyArray_STRIDES(nd_pvals);
	npy_intp pval_s = PyArray_SIZE(nd_pvals);
	npy_intp pval_st_el[pval_ndim];
	for (int i=0; i<pval_ndim; i++) {
		pval_st_el[i] = pval_st_bytes[i] / sizeof(double);
	}
	// get pointers
	double *poi = (double *)PyArray_DATA(nd_poi);
	double *pvals = (double *)PyArray_DATA(nd_pvals);
	
	// create output arrays: LL and ULL for each toy.  two 1d arrays
	//npy_intp const Odims[1];
	//Odims[0] = nTOYS;
	//npy_intp *Odims = &nTOYS;
	npy_intp Odims[1];
	Odims[0] = nTOYS;
	PyObject *nd_LL = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	PyObject *nd_UL = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	PyObject *nd_maxpoi = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	double *LL = (double *)PyArray_DATA(nd_LL);
	double *UL = (double *)PyArray_DATA(nd_UL);
	double *maxpoi = (double *)PyArray_DATA(nd_maxpoi);
	
	double pMax;
	npy_intp i_pMax;
	npy_intp ip;
	npy_intp rj = pval_st_el[0]; // 'rj' for 'row jump'
	npy_intp ipm1;
	//pvals[poi, toy] -> pvals[poi*pval_st_el + toy]
	for (npy_intp i_toy=0; i_toy<nTOYS; i_toy++) {
		pMax = -100.;
		i_pMax = 0;
		// first, find the max val and position/index
		for (ip=0; ip<nPOI; ip++) {
			if (pvals[ip*rj+i_toy] > pMax) {
				pMax = pvals[ip*rj+i_toy];
				i_pMax = ip;
			}
		}
		maxpoi[i_toy] = poi[i_pMax];
		// find the UL
		if (pvals[(nPOI-1)*rj+i_toy] >= p0) {
			UL[i_toy] = poi[(nPOI-1)];
		}
		else {
			ip = i_pMax;
			while ((pvals[ip*rj+i_toy] > p0) && (ip<nPOI) && ((ip*rj+i_toy)<pval_s)){
				ip++;
			}
			// Linear interpolation step
			UL[i_toy] = poi[ip-1] + (p0 - pvals[(ip-1)*rj+i_toy]) * 
				(poi[ip] - poi[ip-1]) / (pvals[ip*rj+i_toy] - pvals[(ip-1)*rj+i_toy]);
		}
		// find the LL
		if (pvals[0*rj+i_toy] >= p0) {
			LL[i_toy] = poi[0];
		}
		else {
			ip = 0;
			while ((pvals[ip*rj+i_toy] < p0) && (ip<i_pMax) && ((ip*rj+i_toy)<pval_s)) {
				ip++;
			}
			LL[i_toy] = poi[ip-1] + (p0 - pvals[(ip-1)*rj+i_toy]) * 
				(poi[ip] - poi[ip-1]) / (pvals[ip*rj+i_toy] - pvals[(ip-1)*rj+i_toy]);
		}
	}
	// release reference to input arrays
	Py_DECREF(nd_poi);
	Py_DECREF(nd_pvals);
	
	// create output tuple
	PyObject *out_tuple =  PyTuple_New(3);
	PyTuple_SetItem(out_tuple, 0, nd_LL);
	PyTuple_SetItem(out_tuple, 1, nd_UL);
	PyTuple_SetItem(out_tuple, 2, nd_maxpoi);
	return out_tuple;
}

// Intended signature: plr.fit_nu(Nb[kpoi,:],Ns[kpoi,:],f_is,f_ib,g_is,g_ib,nu_b)
static PyObject *meth_fit_mu(PyObject *self, PyObject *args) {
	PyArrayObject *nd_Nb, *nd_Ns, *nd_f_is, *nd_f_ib, *nd_g_is, *nd_g_ib;
	double mu_b;
	if (!PyArg_ParseTuple(args, "O&O&O&O&O&O&d", 
		PyArray_Converter, &nd_Nb,
		PyArray_Converter, &nd_Ns,
		PyArray_Converter, &nd_f_is,
		PyArray_Converter, &nd_f_ib,
		PyArray_Converter, &nd_g_is,
		PyArray_Converter, &nd_g_ib,
		&mu_b)) {
		return NULL;
	}
	npy_intp nsim = PyArray_SIZE(nd_Nb);
	long *Nb = (long *)PyArray_DATA(nd_Nb);
	long *Ns = (long *)PyArray_DATA(nd_Ns);
	double *f_is = (double *)PyArray_DATA(nd_f_is);
	double *f_ib = (double *)PyArray_DATA(nd_f_ib);
	double *g_is = (double *)PyArray_DATA(nd_g_is);
	double *g_ib = (double *)PyArray_DATA(nd_g_ib);
	
	npy_intp *Odims = &nsim;
	PyObject *nd_mu_hat = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	PyObject *nd_dL_est = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	double *mu_hat = (double *)PyArray_DATA(nd_mu_hat);
	double *dL_est = (double *)PyArray_DATA(nd_dL_est);
	
	npy_intp ixs=0;
	npy_intp ixs_next;
	npy_intp ixb=0;
	npy_intp ixb_next;
	double mu_hat_est;
	double mu_l, mu_r; // left and right values of nu_signal
	double dL_l, dL_r; // dlogL evaluated at nu_l and nu_r
	double dL_mid;
	npy_intp num_iter;
	for (npy_intp i0=0; i0<nsim; i0++) {
		ixs_next = ixs + Ns[i0];
		ixb_next = ixb + Nb[i0];
		mu_l = 0.0000001;
		mu_r = (double)Ns[i0] * 3. + 10.;
		dL_l = dlogL(mu_l, mu_b, ixs, ixs_next, ixb, ixb_next, f_is, g_is, f_ib, g_ib);
		dL_r = dlogL(mu_r, mu_b, ixs, ixs_next, ixb, ixb_next, f_is, g_is, f_ib, g_ib);
		if ((dL_l <= 0.) && (dL_r < 0.)) {
			mu_hat[i0] = 0.;
			dL_est[i0] = dL_l;
		}
		else if ((dL_l > 0.) && (dL_r > 0.)) {
			mu_hat[i0] = mu_r;
			dL_est[i0] = dL_r;
		}
		else if ((dL_l < 0.) && (dL_r > 0.)) {
			if (dL_l >= dL_r) {
				mu_hat[i0] = mu_l;
				dL_est[i0] = dL_l;
			}
			else {
				mu_hat[i0] = mu_r;
				dL_est[i0] = dL_r;
			}
		}
		else {
			mu_hat_est = 0.5 * (mu_l + mu_r);
			num_iter = 0;
			while (((mu_r - mu_l) > 0.0001) && (num_iter < 1000)) {
				dL_mid = dlogL(mu_hat_est,mu_b,ixs,ixs_next,ixb,ixb_next,f_is,g_is,f_ib,g_ib);
				if (dL_mid >= 0.) {
					mu_l = mu_hat_est;
					dL_l = dL_mid;
				}
				else {
					mu_r = mu_hat_est;
					dL_r = dL_mid;
				}
				mu_hat_est = 0.5 * (mu_l + mu_r);
				num_iter++;
			}
			mu_hat[i0] = mu_hat_est;
			dL_est[i0] = dlogL(mu_hat_est,mu_b,ixs,ixs_next,ixb,ixb_next,f_is,g_is,f_ib,g_ib);
		}
		ixs = ixs_next;
		ixb = ixb_next;
	}
	
	Py_DECREF(nd_Nb);
	Py_DECREF(nd_Ns);
	Py_DECREF(nd_f_is);
	Py_DECREF(nd_g_is);
	Py_DECREF(nd_f_ib);
	Py_DECREF(nd_g_ib);
	
	// Create output tuple and fill with appropriate arrays
	PyObject *out_tuple = PyTuple_New(2);
	PyTuple_SetItem(out_tuple, 0, nd_mu_hat); // estimate of mu-hat
	PyTuple_SetItem(out_tuple, 1, nd_dL_est); // derivative of log-likelihood at mu-hat estimate
	//return nd_mu_hat;
	return out_tuple;
}

static PyObject *meth_log_L(PyObject *self, PyObject *args) {
	// log_L(mu_s, mu_b, Ns, Nb, f_is, f_ib, g_is, g_ib)
	double mu_s, mu_b;
	PyArrayObject *nd_Ns, *nd_Nb, *nd_f_is, *nd_f_ib, *nd_g_is, *nd_g_ib;
	if (!PyArg_ParseTuple(args, "ddO&O&O&O&O&O&", 
		&mu_s,
		&mu_b,
		PyArray_Converter, &nd_Ns,
		PyArray_Converter, &nd_Nb,
		PyArray_Converter, &nd_f_is,
		PyArray_Converter, &nd_f_ib,
		PyArray_Converter, &nd_g_is,
		PyArray_Converter, &nd_g_ib)) {
		return NULL;
	}
	npy_intp nsim = PyArray_SIZE(nd_Nb);
	long *Nb = (long *)PyArray_DATA(nd_Nb);
	long *Ns = (long *)PyArray_DATA(nd_Ns);
	double *f_is = (double *)PyArray_DATA(nd_f_is);
	double *f_ib = (double *)PyArray_DATA(nd_f_ib);
	double *g_is = (double *)PyArray_DATA(nd_g_is);
	double *g_ib = (double *)PyArray_DATA(nd_g_ib);
	
	npy_intp *Odims = &nsim;
	PyObject *nd_logL = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	double *logL = (double *)PyArray_DATA(nd_logL);
	npy_intp is0 = 0;
	npy_intp ib0 = 0;
	npy_intp is_next, ib_next;
	for (npy_intp isim=0;isim<nsim;isim++) {
		is_next = is0 + Ns[isim];
		ib_next = ib0 + Nb[isim];
		logL[isim] = f_logL(mu_s,mu_b,is0,is_next,ib0,ib_next,f_is,g_is,f_ib,g_ib);
		//double f_logL(mus,mub,is0,is1,ib0,ib1,*fs_ii,*gs_ii,*fb_ii,*gb_ii) 
		is0 = is_next;
		ib0 = ib_next;
	}
	Py_DECREF(nd_Nb);
	Py_DECREF(nd_Ns);
	Py_DECREF(nd_f_is);
	Py_DECREF(nd_f_ib);
	Py_DECREF(nd_g_is);
	Py_DECREF(nd_g_ib);
	
	return nd_logL;
}
static PyObject *meth_log_Lhat(PyObject *self, PyObject *args) {
	// log_Lhat(mu_s, mu_b, Ns, Nb, f_is, f_ib, g_is, g_ib)
	double mu_b;
	PyArrayObject *nd_mu_s, *nd_Ns, *nd_Nb, *nd_f_is, *nd_f_ib, *nd_g_is, *nd_g_ib;
	if (!PyArg_ParseTuple(args, "O&dO&O&O&O&O&O&", 
		PyArray_Converter, &nd_mu_s,
		&mu_b,
		PyArray_Converter, &nd_Ns,
		PyArray_Converter, &nd_Nb,
		PyArray_Converter, &nd_f_is,
		PyArray_Converter, &nd_f_ib,
		PyArray_Converter, &nd_g_is,
		PyArray_Converter, &nd_g_ib)) {
		return NULL;
	}
	npy_intp nsim = PyArray_SIZE(nd_Nb);
	double *mu_s = (double *)PyArray_DATA(nd_mu_s);
	long *Nb = (long *)PyArray_DATA(nd_Nb);
	long *Ns = (long *)PyArray_DATA(nd_Ns);
	double *f_is = (double *)PyArray_DATA(nd_f_is);
	double *f_ib = (double *)PyArray_DATA(nd_f_ib);
	double *g_is = (double *)PyArray_DATA(nd_g_is);
	double *g_ib = (double *)PyArray_DATA(nd_g_ib);
	
	npy_intp *Odims = &nsim;
	PyObject *nd_logL = PyArray_EMPTY(1, Odims, NPY_DOUBLE, NPY_CORDER);
	double *logL = (double *)PyArray_DATA(nd_logL);
	npy_intp is0 = 0;
	npy_intp ib0 = 0;
	for (npy_intp isim=0;isim<nsim;isim++) {
		logL[isim] = f_logL(mu_s[isim],mu_b,is0,is0+Ns[isim],ib0,ib0+Nb[isim],f_is,g_is,f_ib,g_ib);
		is0 += Ns[isim];
		ib0 += Nb[isim];
	}
	Py_DECREF(nd_mu_s);
	Py_DECREF(nd_Nb);
	Py_DECREF(nd_Ns);
	Py_DECREF(nd_f_is);
	Py_DECREF(nd_f_ib);
	Py_DECREF(nd_g_is);
	Py_DECREF(nd_g_ib);
	
	return nd_logL;
}

static PyObject *meth_p_vals_ordered(PyObject *self, PyObject *args, PyObject *kwargs) {
	//PyArrayObject *nd_poi, *nd_pvals;
	char *direction = "right";
	PyArrayObject *nd_a, *nd_score;
	static char *keywords[] = {"", "", "direction", NULL};
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O&O&|s", keywords,
		PyArray_Converter, &nd_a,
		PyArray_Converter, &nd_score,
		&direction)) {
		return NULL;
	}
	npy_intp num_a = PyArray_SIZE(nd_a);
	npy_intp num_score = PyArray_SIZE(nd_score);
	double *a = (double *)PyArray_DATA(nd_a);
	double *score = (double *)PyArray_DATA(nd_score);
	
	PyObject *nd_pvals = PyArray_NewLikeArray(nd_score, NPY_ANYORDER, NULL, 1);
	double *pvals = (double *)PyArray_DATA(nd_pvals);
	npy_intp ac = 0; // a count
	if (strcmp(direction,"right")==0) {  //0 means it matches
		//printf("pvals are right\n");
		for (npy_intp i=0;i<num_score;i++) {
			while ((a[ac]<score[i])&&(ac<num_a)) {
				ac++;
			}
			pvals[i] = ((double)(num_a-ac)) / ((double)num_a);
		}
	} else if (strcmp(direction,"left")==0) {
		//printf("pvals are left\n");
		for (npy_intp i=0;i<num_score;i++) {
			while ((a[ac]<=score[i])&&(ac<num_a)) {
				ac++;
			}
			pvals[i] = ((double)ac) / ((double)num_a);
		}
	} else {
		PyErr_SetString(PyExc_ValueError,"kwarg 'direction' must be either 'left' or 'right'");
		Py_DECREF(nd_a);
		Py_DECREF(nd_score);
		Py_DECREF(nd_pvals);
		return NULL;
	}
	Py_DECREF(nd_a);
	Py_DECREF(nd_score);
	//Py_DECREF(direction);
	return nd_pvals;
}
// to unsort: a_sorted[a.argsort().argsort()]
/* ----------------- </MODULE FUNCTIONS> ----------------- */

/* ----------------- <DOC STRINGS> ----------------- */
PyDoc_STRVAR(
	get_limits__doc__,
	"get_limits(poi, p_vals, p0=0.1)\n--\n\n"
	"p_vals is a (num_poi x num_toys) array\n"
	"poi should be a 1d array, length num_poi\n"
	"p0 should be a float64/double between 0 and 1.  p0=0.1 by default\n"
	"returns three arrays (as a tuple): LL, UL, maxPoi (in that order)");

// Intended signature: plr.fit_mu(Nb[kpoi,:],Ns[kpoi,:],f_is,f_ib,g_is,g_ib,nu_b)
PyDoc_STRVAR(
	fit_mu__doc__,
	"fit_mu(Nb,Ns,f_is,f_ib,g_is,g_ib,mu_b)\n--\n\n"
	"Nb and Ns are 1D arrays, for a single poi (one element per toy).\n"
	"f_is and f_ib are the values of the signal PDF evaluated at the\n"
	"    signal and background x-values, respectively.\n"
	"g_is and g_ib are the values of the background PDF evaluated at the\n"
	"    signal and background x-values, respectively.\n"
	"len(f_is) and len(g_is) should be equal to Ns.sum(),\n"
	"len(f_ib) and len(g_ib) should be equal to Nb.sum().\n"
	"mu_b is a scalar double: the true background expectation value.\n"
	"Returns:\n"
	"    mu-hat: array with mu-hat values\n"
	"      dLog: array of values of the log-likelihood derivative at mu-hat. This\n"
	"            will be close to zero (condition of extrema), except when the\n"
	"            preferred mu-hat is negative.  In this case, mu-hat=0 and the\n"
	"            derivative will *not* necessarily be zero.");
PyDoc_STRVAR(
	log_L__doc__,
	"log_L(mu_s, mu_b, Ns, Nb, f_is, f_ib, g_is, g_ib)\n--\n\n"
	"mu_s is the (scalar, float) signal hypothesis.\n"
	"mu_b is the (scalar, float) background hypothesis.\n"
	"Ns, Nb, are both 1D numpy arrays, same size (length: number of toy sims;\n"
	"    They describe the number of signal and number of background events,\n"
	"    respectively, per toy sim.\n"
	"f_is, f_ib are 1D: the signal PDF evaluated at the signal and background\n"
	"    x-values, respectively (length: number of toy sims)\n"
	"g_is, g_ib are 1D: the background PDF evaluated at the signal and background\n"
	"    x-values, respetively (length: number of toy sims)");
PyDoc_STRVAR(
	log_Lhat__doc__,
	"log_Lhat(mu_s, mu_b, Ns, Nb, f_is, f_ib, g_is, g_ib)\n--\n\n"
	"mu_s is the 1D array of signal hypotheses (length: number of toy sims).\n"
	"mu_b is the (scalar, float) background hypothesis.\n"
	"Ns, Nb, are both 1D numpy arrays, same size (length: number of toy sims;\n"
	"    They describe the number of signal and number of background events,\n"
	"    respectively, per toy sim.\n"
	"f_is, f_ib are 1D: the signal PDF evaluated at the signal and background\n"
	"    x-values, respectively (length: number of toy sims)\n"
	"g_is, g_ib are 1D: the background PDF evaluated at the signal and background\n"
	"    x-values, respetively (length: number of toy sims)");

PyDoc_STRVAR(
	p_vals_ordered__doc__,
	"p_vals_ordered(a, score, direction='right')\n--\n\n"
	"For every element of 'score', the p-value is evaluated based on the elements\n"
	"found in 'a'.  INPUTS 'a' AND 'score' MUST BE SORTED FROM LOWEST TO HIGHEST.\n"
	"kwarg 'direction' determines whether the p-value is calculated on the left\n"
	"(low) tail or right (high) tail.  That is:\n"
	" direction='right':\n"
	"    pval[i] = eff(a >= score[i])\n"
	" direction='left':\n"
	"    pval[i] = eff(a <= score[i])");
/* ----------------- </DOC STRINGS> ----------------- */

/* ----------------- <BOILER PLATE CODE BELOW THAT BUILDS THE MODULE> ----------------- */
/* module functions are added in this array with a struct of 
	{
		<name as it will appear in python>, 
		<pointer to C function definition>, 
		<Python.h descriptor of what the function will be accepting>, 
		<docstring>
	}
*/
static PyMethodDef Plr_Methods[] = {
	{"get_limits",(PyCFunction)meth_get_limits,METH_VARARGS|METH_KEYWORDS,get_limits__doc__},
	{"fit_mu",meth_fit_mu,METH_VARARGS,fit_mu__doc__},
	{"log_L",meth_log_L,METH_VARARGS,log_L__doc__},
	{"log_Lhat",meth_log_Lhat,METH_VARARGS,log_Lhat__doc__},
	{"p_vals_ordered",(PyCFunction)meth_p_vals_ordered,METH_VARARGS|METH_KEYWORDS,p_vals_ordered__doc__},
	{NULL, NULL, 0, NULL}
};

/* the following struct defines properties of the module itself */
static struct PyModuleDef plr_module = {
	PyModuleDef_HEAD_INIT,
	"profile_likelihood",
	"Low-level functions for analyzing PLR output data",
	-1,
	Plr_Methods
};

/* NOTE: in the function below, 'import_array()' must be included, which does not exist in the other
   other examples that use python-only API functions and variable types.

The name of the function of type PyMODINIT_FUNC has to be "PyInit_{name}", where {name} is the name
of the module as it will be imported from python, and has to match the secend element of the module
struct defined above.
 */
PyMODINIT_FUNC PyInit_profile_likelihood(void) {
	import_array();
    return PyModule_Create(&plr_module);
}
