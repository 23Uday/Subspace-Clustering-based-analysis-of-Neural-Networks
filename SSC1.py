import numpy as np
import pdb

def shrinkageOperator(M,eps):
	ANS = np.zeros(M.shape)
	ANS[M > eps] = M[M>eps] - eps
	ANS[M < -eps] = M[M < -eps] + eps

	return ANS

def fit(X,C):
	return (np.linalg.norm(X-X@C)**2/X.size)**0.5


def ADMMLASSOSSC(X,tau,mu=100,maxIter=100,err = 1e-2):
	print("Function Called")
	d,n = X.shape
	C = np.zeros((n,n))
	C_prev = C
	C_viable = C
	lam2 = np.zeros(C.shape)
	mu2 = mu # This might have to be tuned
	I = np.eye(n)

	factor = 10
	tauI = tauD = 2

	tauXTX = tau*X.T@X
	Linv = np.linalg.pinv(tauXTX + mu2*I)

	itr = 0
	reset = 0
	fitOld = fitnew = fitdelta = 0
	fitnew = fit(X,C)
	fitdelta = fitnew - fitOld

	print("Itr %d| fit %f| delta %f"%(itr,fit(X,C),abs(fitdelta)))

	# Z = Linv@(tauXTX+mu2*(C-lam2/mu2))
	# if 1/mu2 > np.max(Z) or -1/mu2 < np.min(Z):
	# 	print("Warning! 1/mu2 exceeds max or falls below min")
	# 	print("Setting mu2 = 1/np.mean(Z)")
	# 	mu2 = 1/np.mean(Z)
	# pdb.set_trace()
	while fit(X,C) > err and itr < maxIter and reset < 2:
		fitOld = fitnew

		Z = Linv@(tauXTX+mu2*(C-lam2/mu2))
		# Z = Z - np.diag(Z.diagonal())
		# if 1/mu2 > np.max(Z) or -1/mu2 < np.min(Z):
		# print("Max value in Z Matrix: %s"%np.max(Z))
		# print("Min value in Z Matrix: %s\n"%np.min(Z))

			# print("Warning! 1/mu2 exceeds max or falls below min")
			# print("Setting mu2 = 1/np.mean(Z)\n")
			# mu2 = np.abs(1/np.min(Z))
		# pdb.set_trace()
		C_prev = C

		C = shrinkageOperator(Z + lam2/mu2, 1/mu2)
		C = C - np.diag(C.diagonal())


		
		# pdb.set_trace()
		# print("Max value in C Matrix: %s"%np.max(C))
		# print("Min value in C Matrix: %s\n"%np.min(C))

		lam2 += mu2 * (Z - C)

		itr += 1

		# pdb.set_trace()
		# Stephen Boyd Huersitic
		S = mu2*(C-C_prev)
		R = Z - C + np.diag(C.diagonal())

		if np.linalg.norm(R) > factor * np.linalg.norm(S):
			mu2 *= tauI
			# print("mu2 increased : %s"%mu2)
		elif np.linalg.norm(R) * factor < np.linalg.norm(S):
			mu2 /= tauD
			# print("mu2 decreased : %s"%mu2)
		

		fitnew = fit(X,C)
		fitdelta = fitnew - fitOld

		if C.max() > 0 or C.min() < 0 and fitdelta < 0:
			C_viable = C

		print("Itr %d| fit %f| delta %f| mu2 %s"%(itr,fit(X,C),abs(fitdelta),mu2))

		if itr == maxIter - 1:
			if fit(X,C) > 1e-1:
				maxIter *= 2
				reset += 1
	
	if C.max() == 0:
		print("Returning a Viable C instead of Optimal C")
		C = C_viable
		print("Fit of the returned C : %s"%fit(X,C))



	return C

def affinityMatrix(C):
	absC = np.abs(C)
	return absC + absC.T