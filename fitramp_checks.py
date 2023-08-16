import numpy as np
from scipy import optimize
import fitramp


def run_checks(nreads_Cmatrix=1000000, countrate=20, sig=20, nramps_test=10,
               nramps_testjumps=100000, 
               readtimes=[1, 2, 3, [4, 5], [6, 7, 8], [9, 10, 12], [14, 17],
                          [20, 21], 22, [24, 25]]):

    """

    Run a bunch of checks.  This will be converted into unit tests.

    """
    
    allpassed = True
    countrate = 20
    sig = 20
    
    C = fitramp.Covar(readtimes, pedestal=True)
    
    print("Generating sample ramps")
    y = fitramp.getramps(countrate, sig, readtimes, nramps=nreads_Cmatrix)
    d = y*1.
    d[0] /= C.mean_t[0]
    d[1:] = (y[1:] - y[:-1])/C.delta_t[:, np.newaxis]
    # the line below is without a pedestal fit
    #d = (y[1:] - y[:-1])/C.delta_t[:, np.newaxis]
    
    print("\nEmpirically checking the covariance matrix for read pattern", readtimes)
    print("Diagonal elements ...")
    
    print("Empirical: ", np.var(d, axis=1).round(2))
    print("Calculated:", (countrate*C.alpha_phnoise + sig**2*C.alpha_readnoise).round(2))
    
    print("Nonzero off-diagonal elements ...")
    meanref = np.mean(d[1:], axis=1)*np.mean(d[:-1], axis=1)
    print("Empirical: ", (np.mean(d[1:]*d[:-1], axis=1) - meanref).round(2))
    print("Calculated:", (countrate*C.beta_phnoise + sig**2*C.beta_readnoise).round(2))

    print("Hopefully zero off-diagonal elements ...")
    for i in range(2, d.shape[0]):
        meanref = np.mean(d[i:], axis=1)*np.mean(d[:-i], axis=1)
        print("Empirical: ", (np.mean(d[i:]*d[:-i], axis=1) - meanref).round(2))

    # Test the pedestal fitting here.

    print("\nChecking the pedestal fit")
    
    C = fitramp.Covar(readtimes, pedestal=True)
    
    # First "difference" is actually the (scaled) first resultant.
    y = fitramp.getramps(countrate, sig, readtimes, nramps=nramps_test)
    d = y*1.
    d[0] /= C.mean_t[0]
    d[1:] = (y[1:] - y[:-1])/C.delta_t[:, np.newaxis]
    r = fitramp.fit_ramps(d, C, sig)

    # Construct the explicit covariance matrix, use curve_fit, and
    # check that the parameters and covariance matrix are correctly
    # recovered.  Use curve_fit for the test.

    M = np.zeros((d.shape[0], d.shape[0]))
    ct = np.mean(d[1:, 0])
    for i in range(d.shape[0]):
        M[i, i] = C.alpha_readnoise[i]*sig**2 + C.alpha_phnoise[i]*ct
        if i < d.shape[0] - 1:
            M[i, i + 1] = M[i + 1, i] = C.beta_readnoise[i]*sig**2 + C.beta_phnoise[i]*ct
            
    def lin(x, a, b):
        val = x*0 + a
        val[0] += b
        return val

    par, cov = optimize.curve_fit(lin, np.arange(d.shape[0]), d[:, 0], sigma=M, absolute_sigma=True)
    par_ramp = np.array([r.countrate[0], r.pedestal[0]])
    cov_ramp = np.array([[r.uncert[0]**2, r.covar_countrate_pedestal[0]],
                          [r.covar_countrate_pedestal[0], r.uncert_pedestal[0]**2]])
    
    if not np.all(np.isclose(par, par_ramp)) or not np.all(np.isclose(cov, cov_ramp)):
        print("Failed initial pedestal test.")
        allpassed = False

    # Now repeat the test with a missing resultant difference.  The
    # explicit covariance matrix will have a slightly smaller
    # dimension and no off-diagonal terms at the now-decoupled
    # resultant differences.
    
    # difference to omit: choose one near the middle.
    ii = d.shape[0]//2
    
    diffs2use = np.ones(d.shape)
    diffs2use[ii] = 0
    
    Mp = np.zeros((d.shape[0] - 1, d.shape[0] - 1))
    ct = np.mean(d[1:, 0])
    for i in range(d.shape[0]):
        if i < ii:
            Mp[i, i] = C.alpha_readnoise[i]*sig**2 + C.alpha_phnoise[i]*ct
            if i < d.shape[0] - 1:
                Mp[i, i + 1] = Mp[i + 1, i] = C.beta_readnoise[i]*sig**2 + C.beta_phnoise[i]*ct
        elif i > ii:
            Mp[i - 1, i - 1] = C.alpha_readnoise[i]*sig**2 + C.alpha_phnoise[i]*ct
            if i < d.shape[0] - 1:
                Mp[i - 1, i] = Mp[i, i - 1] = C.beta_readnoise[i]*sig**2 + C.beta_phnoise[i]*ct

    Mp[ii, ii - 1] = Mp[ii - 1, ii] = 0

    # Difference vector with the bad difference omitted.
    
    dp = np.zeros((d.shape[0] - 1, d.shape[1]))
    dp[:ii] = d[:ii]
    dp[ii:] = d[ii + 1:]
    
    r = fitramp.fit_ramps(d, C, sig, countrateguess=np.mean(d[1:], axis=0), diffs2use=diffs2use)

    par, cov = optimize.curve_fit(lin, np.arange(dp.shape[0]), dp[:, 0], sigma=Mp, absolute_sigma=True)
    par_ramp = np.array([r.countrate[0], r.pedestal[0]])
    cov_ramp = np.array([[r.uncert[0]**2, r.covar_countrate_pedestal[0]],
                          [r.covar_countrate_pedestal[0], r.uncert_pedestal[0]**2]])
    
    if not np.all(np.isclose(par, par_ramp)) or not np.all(np.isclose(cov, cov_ramp)):
        print("Failed pedestal test with a masked difference.")
        allpassed = False

    #
    # Use just a few test ramps; no need for many ramps here.
    #
    
    C = fitramp.Covar(readtimes)
    y = fitramp.getramps(countrate, sig, readtimes, nramps=nramps_test)
    d = (y[1:] - y[:-1])/C.delta_t[:, np.newaxis]
    
    countrateguess = np.mean(d, axis=0)[np.newaxis, :]
    countrateguess *= countrateguess > 0
    alpha = countrateguess*C.alpha_phnoise[:, np.newaxis]
    alpha += sig**2*C.alpha_readnoise[:, np.newaxis]
    beta = countrateguess*C.beta_phnoise[:, np.newaxis]
    beta += sig**2*C.beta_readnoise[:, np.newaxis]
    result = fitramp.fit_ramps(d, C, sig, detect_jumps=True)
    
    # Now perform several checks to make sure that I get the same
    # answer from using the auxiliary variables as from inverting the
    # covariance matrix.  Do the tests omitting several reads too to
    # ensure that this works.  Any disagreement between the best-fit
    # count rate, its uncertainty, or chi squared of 1e-8 or larger is
    # failure.

    print("\nChecking results for the full ramp")
    
    for i in range(nramps_test):
        Covar = np.zeros((d.shape[0], d.shape[0]))
        for j in range(d.shape[0]):
            Covar[j, j] = alpha[j, i]
        for j in range(d.shape[0] - 1):
            Covar[j, j + 1] = Covar[j + 1, j] = beta[j, i]

        Cinv = np.linalg.inv(Covar)
        slope = np.sum(d[:, i]*Cinv)/np.sum(Cinv)
        chisq = np.linalg.multi_dot([(d[:, i] - slope)[np.newaxis, :], Cinv, d[:, i] - slope])[0]

        if np.abs(slope - result.countrate[i]) > 1e-8 or \
           np.abs(chisq - result.chisq[i]) > 1e-8 or \
           np.abs(np.sum(Cinv)**-0.5 - result.uncert[i]) > 1e-8:
            print("Failed test for full ramp")
            allpassed = False
    
    # Make sure that the answer is correct omitting one read in turn
    # on the first ramp

    Covar_saved = np.zeros((d.shape[0], d.shape[0]))
    for j in range(d.shape[0]):
        Covar_saved[j, j] = alpha[j, 0]
    for j in range(d.shape[0] - 1):
        Covar_saved[j, j + 1] = Covar_saved[j + 1, j] = beta[j, 0]
    
    print("Checking results for one omitted read")
    
    for i in range(d.shape[0]):
        Covar = Covar_saved*1.
        
        # read to omit is i
        Covar[i, i] = 1e20
        if i < d.shape[0] - 1:
            Covar[i + 1, i] = Covar[i, i + 1] = 0
        if i > 0:
            Covar[i - 1, i] = Covar[i, i - 1] = 0
        
        Cinv = np.linalg.inv(Covar)
        slope = np.sum(d[:, 0]*Cinv)/np.sum(Cinv)
        chisq = np.linalg.multi_dot([(d[:, 0] - slope)[np.newaxis, :], Cinv, d[:, 0] - slope])[0]
        
        if np.abs(slope - result.countrate_oneomit[i, 0]) > 1e-8 or \
           np.abs(chisq - result.chisq_oneomit[i, 0]) > 1e-8 or \
           np.abs(np.sum(Cinv)**-0.5 - result.uncert_oneomit[i, 0]) > 1e-8:
            allpassed = False
            print("Failed test for one omitted resultant difference")    
    
    print("Checking results with two omitted reads")
    
    for i in range(d.shape[0] - 1):
        Covar = Covar_saved*1.
    
        # reads to omit are i, i+1
        Covar[i, i] = 1e20
        Covar[i + 1, i + 1] = 1e20
    
        Covar[i + 1, i] = Covar[i, i + 1] = 0
                
        if i < d.shape[0] - 2:
            Covar[i + 2, i] = Covar[i, i + 2] = 0
        if i > 0:
            Covar[i - 1, i] = Covar[i, i - 1] = 0

        Cinv = np.linalg.inv(Covar)
        slope = np.sum(d[:, 0]*Cinv)/np.sum(Cinv)
        chisq = np.linalg.multi_dot([(d[:, 0] - slope)[np.newaxis, :], Cinv, d[:, 0] - slope])[0]
        
        if np.abs(slope - result.countrate_twoomit[i, 0]) > 1e-8 or \
           np.abs(chisq - result.chisq_twoomit[i, 0]) > 1e-8 or \
           np.abs(np.sum(Cinv)**-0.5 - result.uncert_twoomit[i, 0]) > 1e-8:
            allpassed = False
            print("Failed test for two omitted resultant differences")
    
    print("Checking three random omitted reads")

    diffs2use = np.ones(d.shape)
    Covar = Covar_saved*1.

    for j in range(3):
        i = np.random.randint(d.shape[0])
        diffs2use[i] = 0
        
        # omit read i
        
        Covar[i, i] = 1e20
                
        if i < d.shape[0] - 1:
            Covar[i + 1, i] = Covar[i, i + 1] = 0
        if i > 0:
            Covar[i - 1, i] = Covar[i, i - 1] = 0
        
        Cinv = np.linalg.inv(Covar)
        slope = np.sum(d[:, 0]*Cinv)/np.sum(Cinv)
        chisq = np.linalg.multi_dot([(d[:, 0] - slope)[np.newaxis, :], Cinv, d[:, 0] - slope])[0]
        
        result = fitramp.fit_ramps(d, C, sig, detect_jumps=True,
                                   diffs2use=diffs2use,
                                   countrateguess=countrateguess)

    if np.abs(slope - result.countrate[0]) > 1e-8 or \
           np.abs(chisq - result.chisq[0]) > 1e-8 or \
           np.abs(np.sum(Cinv)**-0.5 - result.uncert[0]) > 1e-8:
        allpassed = False
        print("Failed test for three omitted resultant differences")

    Covar_saved_threeomit = Covar*1.

    for i in range(d.shape[0]):
        Covar = Covar_saved_threeomit*1.
    
        Covar[i, i] = 1e20
                
        if i < d.shape[0] - 1:
            Covar[i + 1, i] = Covar[i, i + 1] = 0
        if i > 0:
            Covar[i - 1, i] = Covar[i, i - 1] = 0
        
        Cinv = np.linalg.inv(Covar)
        slope = np.sum(d[:, 0]*Cinv)/np.sum(Cinv)
        chisq = np.linalg.multi_dot([(d[:, 0] - slope)[np.newaxis, :], Cinv, d[:, 0] - slope])[0]
        
        if np.abs(slope - result.countrate_oneomit[i, 0]) > 1e-8 or \
           np.abs(chisq - result.chisq_oneomit[i, 0]) > 1e-8 or \
           np.abs(np.sum(Cinv)**-0.5 - result.uncert_oneomit[i, 0]) > 1e-8:
            allpassed = False
            print("Failed test for three+one omitted resultant differences")

    for i in range(d.shape[0] - 1):
        Covar = Covar_saved_threeomit*1.
    
        Covar[i, i] = Covar[i + 1, i + 1] = 1e20
        Covar[i + 1, i] = Covar[i, i + 1] = 0
        
        if i < d.shape[0] - 2:
            Covar[i + 2, i] = Covar[i, i + 2] = 0
        if i > 0:
            Covar[i - 1, i] = Covar[i, i - 1] = 0
        
        Cinv = np.linalg.inv(Covar)
        slope = np.sum(d[:, 0]*Cinv)/np.sum(Cinv)
        chisq = np.linalg.multi_dot([(d[:, 0] - slope)[np.newaxis, :], Cinv, d[:, 0] - slope])[0]

        if np.abs(slope - result.countrate_twoomit[i, 0]) > 1e-8 or \
           np.abs(chisq - result.chisq_twoomit[i, 0]) > 1e-8 or \
           np.abs(np.sum(Cinv)**-0.5 - result.uncert_twoomit[i, 0]) > 1e-8:
            allpassed = False
            print("Failed test for three+two omitted resultant differences")

    # Now check jump detection.

    print("\nChecking jump detection")
    
    y = fitramp.getramps(countrate, sig, readtimes, nramps=nramps_testjumps)
    d = (y[1:] - y[:-1])/C.delta_t[:, np.newaxis]

    # add a big jump to 2% of the read differences
    add_jump = np.random.rand(*d.shape) < 0.02
    d[add_jump] += (countrate + sig)*8

    sig_arr = sig*np.ones(d.shape[1])
    diffs2use, c = fitramp.mask_jumps(d, C, sig_arr, threshold_oneomit=18,
                                   threshold_twoomit=21)

    # We should successfully detect all jumps unless a row has
    # at least nresultants/2 jumps.

    toomanyjumps = np.sum(add_jump, axis=0) >= add_jump.shape[0]//2

    numberflagged = np.sum((1 - diffs2use)*(~toomanyjumps))
    numbercaught = np.sum(add_jump*(1 - diffs2use)*(~toomanyjumps))
    numbermissed = np.sum(add_jump*diffs2use*(~toomanyjumps))

    # Depending on the readout scheme we may have had to discard extra
    # resultant differences.  Account for that when checking success
    # here.  These lines are copied from the function in fitramp.
    
    oneomit_ok = C.Nreads[1:]*C.Nreads[:-1] == 1
    oneomit_ok[0] = oneomit_ok[-1] = True

    factor_expected = (np.sum(oneomit_ok) + 2*np.sum(~oneomit_ok))/d.shape[0]

    # Success = numberflagged \approx numbercaught*factor_expected,
    # numbermissed \approx zero

    if np.abs(numberflagged/factor_expected/numbercaught - 1) > 0.1:
        print("Jump masking flagged an unexpectedly large number of valid resultant differences")
        allpassed = False
    if numbermissed/numbercaught > 1e-3:
        print("Jump masking missed an unexpectedly large number of jumps.")
        allpassed = False

    # Check that the weights returned are correct discarding a handful
    # of reads

    print("\nChecking weights")

    y = fitramp.getramps(countrate, sig, readtimes, nramps=nramps_testjumps)
    d = (y[1:] - y[:-1])/C.delta_t[:, np.newaxis]

    diffs2use = np.random.rand(*d.shape) > 0.02
    diffs2use[0] = 1 # ensure that there is always at least one valid read

    result = fitramp.fit_ramps(d, C, sig, diffs2use=diffs2use)
    weights_resultants = y*0
    weights_resultants[1:] = result.weights/C.delta_t[:, np.newaxis]
    weights_resultants[:-1] -= result.weights/C.delta_t[:, np.newaxis]

    if np.std(np.sum(weights_resultants*y, axis=0) - result.countrate) > 1e-10:
        print("Weights for resultants look incorrect.")
        allpassed = False
    if np.std(np.sum(result.weights*d, axis=0) - result.countrate) > 1e-10:
        print("Weights for resultant differences look incorrect.")        
        allpassed = False
        
    #print((numberflagged/factor_expected).astype(int), numbercaught)
    #print(numbermissed)
    
    if allpassed:
        print("\nAll tests passed.\n")

    
run_checks()
